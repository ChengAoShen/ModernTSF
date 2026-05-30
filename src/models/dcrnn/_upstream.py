"""Vendored DCRNN architecture.

Vendored/adapted from https://github.com/GestaltCogTeam/BasicTS
(baselines/DCRNN), Apache-2.0.

Changes from upstream:
* ``DCGRUCell._supports`` is supplied as a list of dense ``torch.Tensor``
  transition matrices (built by the adapter from the predefined adjacency).
* All hardcoded ``.cuda()`` / device assumptions removed; every
  internally-created tensor follows the input tensor's device.
* ``LayerParams`` materialises its weights/biases on the correct device the
  first time they are requested.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class LayerParams:
    """Lazily-initialised, registered layer parameters."""

    def __init__(self, rnn_network: nn.Module, layer_type: str):
        self._rnn_network = rnn_network
        self._params_dict: dict = {}
        self._biases_dict: dict = {}
        self._type = layer_type

    def get_weights(self, shape):
        if shape not in self._params_dict:
            nn_param = nn.Parameter(torch.empty(*shape))
            nn.init.xavier_normal_(nn_param)
            self._params_dict[shape] = nn_param
            self._rnn_network.register_parameter(
                "{}_weight_{}".format(self._type, str(shape)), nn_param
            )
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = nn.Parameter(torch.empty(length))
            nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter(
                "{}_biases_{}".format(self._type, str(length)), biases
            )
        return self._biases_dict[length]


class DCGRUCell(nn.Module):
    """Diffusion-convolutional GRU cell.

    ``supports`` is a list of dense ``(N, N)`` transition matrices. They are
    stored as registered buffers so they follow the module's device.
    """

    def __init__(
        self,
        num_units,
        supports,
        max_diffusion_step,
        num_nodes,
        nonlinearity="tanh",
        use_gc_for_ru=True,
    ):
        super().__init__()
        self._activation = torch.tanh if nonlinearity == "tanh" else torch.relu
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._use_gc_for_ru = use_gc_for_ru

        # Register the support matrices as buffers so they follow the device.
        self._num_supports = len(supports)
        for i, support in enumerate(supports):
            self.register_buffer(f"_support_{i}", support, persistent=False)

        self._fc_params = LayerParams(self, "fc")
        self._gconv_params = LayerParams(self, "gconv")

    @property
    def _supports(self):
        return [getattr(self, f"_support_{i}") for i in range(self._num_supports)]

    def forward(self, inputs, hx):
        output_size = 2 * self._num_units
        fn = self._gconv if self._use_gc_for_ru else self._fc
        value = torch.sigmoid(fn(inputs, hx, output_size, bias_start=1.0))
        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(
            tensor=value, split_size_or_sections=self._num_units, dim=-1
        )
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))

        c = self._gconv(inputs, r * hx, self._num_units)
        if self._activation is not None:
            c = self._activation(c)

        new_state = u * hx + (1.0 - u) * c
        return new_state

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def _fc(self, inputs, state, output_size, bias_start=0.0):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)
        input_size = inputs_and_state.shape[-1]
        weights = self._fc_params.get_weights((input_size, output_size)).to(
            inputs_and_state.device
        )
        value = torch.sigmoid(torch.matmul(inputs_and_state, weights))
        biases = self._fc_params.get_biases(output_size, bias_start)
        value += biases.to(inputs_and_state.device)
        return value

    def _gconv(self, inputs, state, output_size, bias_start=0.0):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        state = torch.reshape(state, (batch_size, self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.size(2)

        x = inputs_and_state
        x0 = x.permute(1, 2, 0)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in self._supports:
                x1 = torch.mm(support.to(x0.device), x0)
                x = self._concat(x, x1)
                for _k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * torch.mm(support.to(x0.device), x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        num_matrices = len(self._supports) * self._max_diffusion_step + 1
        x = torch.reshape(
            x, shape=[num_matrices, self._num_nodes, input_size, batch_size]
        )
        x = x.permute(3, 1, 2, 0)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(
            x, shape=[batch_size * self._num_nodes, input_size * num_matrices]
        )
        weights = self._gconv_params.get_weights(
            (input_size * num_matrices, output_size)
        ).to(x.device)
        x = torch.matmul(x, weights)
        biases = self._gconv_params.get_biases(output_size, bias_start).to(x.device)
        x += biases
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])


class Seq2SeqAttrs:
    def __init__(self, supports, **model_kwargs):
        self.supports = supports
        self.max_diffusion_step = int(model_kwargs.get("max_diffusion_step", 2))
        self.cl_decay_steps = int(model_kwargs.get("cl_decay_steps", 1000))
        self.filter_type = model_kwargs.get("filter_type", "laplacian")
        self.num_nodes = int(model_kwargs.get("num_nodes", 1))
        self.num_rnn_layers = int(model_kwargs.get("num_rnn_layers", 1))
        self.rnn_units = int(model_kwargs.get("rnn_units"))
        self.hidden_state_size = self.num_nodes * self.rnn_units


class EncoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, supports, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, supports, **model_kwargs)
        self.input_dim = int(model_kwargs.get("input_dim", 1))
        self.seq_len = int(model_kwargs.get("seq_len"))
        self.dcgru_layers = nn.ModuleList(
            [
                DCGRUCell(
                    self.rnn_units, supports, self.max_diffusion_step, self.num_nodes
                )
                for _ in range(self.num_rnn_layers)
            ]
        )

    def forward(self, inputs, hidden_state=None):
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros(
                (self.num_rnn_layers, batch_size, self.hidden_state_size),
                device=inputs.device,
            )
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state
        return output, torch.stack(hidden_states)


class DecoderModel(nn.Module, Seq2SeqAttrs):
    def __init__(self, supports, **model_kwargs):
        nn.Module.__init__(self)
        Seq2SeqAttrs.__init__(self, supports, **model_kwargs)
        self.output_dim = int(model_kwargs.get("output_dim", 1))
        self.horizon = int(model_kwargs.get("horizon", 1))
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.dcgru_layers = nn.ModuleList(
            [
                DCGRUCell(
                    self.rnn_units, supports, self.max_diffusion_step, self.num_nodes
                )
                for _ in range(self.num_rnn_layers)
            ]
        )

    def forward(self, inputs, hidden_state=None):
        hidden_states = []
        output = inputs
        for layer_num, dcgru_layer in enumerate(self.dcgru_layers):
            next_hidden_state = dcgru_layer(output, hidden_state[layer_num])
            hidden_states.append(next_hidden_state)
            output = next_hidden_state
        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)
        return output, torch.stack(hidden_states)


class DCRNN(nn.Module, Seq2SeqAttrs):
    """
    Paper: Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic
    Forecasting. Link: https://arxiv.org/abs/1707.01926. Venue: ICLR 2018.
    """

    def __init__(self, supports, **model_kwargs):
        super().__init__()
        Seq2SeqAttrs.__init__(self, supports, **model_kwargs)
        self.encoder_model = EncoderModel(supports, **model_kwargs)
        self.decoder_model = DecoderModel(supports, **model_kwargs)
        self.cl_decay_steps = int(model_kwargs.get("cl_decay_steps", 2000))
        self.use_curriculum_learning = bool(
            model_kwargs.get("use_curriculum_learning", False)
        )

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
            self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps)
        )

    def encoder(self, inputs):
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(
                inputs[t], encoder_hidden_state
            )
        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, labels=None, batches_seen=None):
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros(
            (batch_size, self.num_nodes * self.decoder_model.output_dim),
            device=encoder_hidden_state.device,
        )
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol
        outputs = []
        for t in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(
                decoder_input, decoder_hidden_state
            )
            decoder_input = decoder_output
            outputs.append(decoder_output)
            if self.training and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self._compute_sampling_threshold(batches_seen):
                    decoder_input = labels[t]
        outputs = torch.stack(outputs)
        return outputs

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor = None,
        batch_seen: int = None,
        **kwargs,
    ) -> torch.Tensor:
        """Feedforward of DCRNN.

        Args:
            history_data: history with shape ``(B, L, N, C)``.
            future_data: optional future with shape ``(B, L, N, C_out)`` for
                teacher forcing.
            batch_seen: batches seen so far (curriculum learning).

        Returns:
            prediction with shape ``(B, L, N, C_out)``.
        """
        batch_size, length, num_nodes, channels = history_data.shape
        history_data = history_data.reshape(batch_size, length, num_nodes * channels)
        history_data = history_data.transpose(0, 1)  # [L, B, N*C]

        if future_data is not None:
            future_data = future_data[..., [0]]
            b2, l2, n2, c2 = future_data.shape
            future_data = future_data.reshape(b2, l2, n2 * c2)
            future_data = future_data.transpose(0, 1)  # [L, B, N*C]

        encoder_hidden_state = self.encoder(history_data)
        outputs = self.decoder(
            encoder_hidden_state, future_data, batches_seen=batch_seen
        )  # [L, B, N*C_out]

        L, B, _ = outputs.shape
        outputs = outputs.transpose(0, 1)  # [B, L, N*C_out]
        outputs = outputs.view(
            B, L, self.num_nodes, self.decoder_model.output_dim
        )
        return outputs
