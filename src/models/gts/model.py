"""ModernTSF adapter for the GTS spatiotemporal graph forecasting model.

Vendored/adapted from https://github.com/GestaltCogTeam/BasicTS (baselines/GTS),
Apache-2.0. GTS (ICLR 2021, "Discrete Graph Structure Learning for Forecasting
Multiple Time Series", https://arxiv.org/abs/2101.06861) learns a discrete graph
structure from a small per-node feature series via a Gumbel-Softmax sampler, then
runs DCRNN-style diffusion-convolutional recurrence over the learned graph.

This adapter:

* converts ModernTSF's ``(x_enc, x_mark_enc)`` into the BasicTS spatiotemporal
  layout ``(B, L, N, 1 + F)`` via :func:`to_spatiotemporal` (channel 0 = value,
  then calendar ``[time_in_day, day_in_week]``);
* drives the upstream ``GTS`` module with the BasicTS forward signature
  ``forward(history_data, future_data, batch_seen, epoch, train)``;
* squeezes the upstream ``(B, pred_len, N, 1)`` prediction back to
  ``(B, pred_len, N)``.

All hardcoded ``cuda`` placements from upstream are removed; internally created
tensors (adjacency, identity, masks, node features) follow the input device. The
graph-structure-learning feature series is generated internally as a fixed buffer
(no external feature file required); the optional KL prior graph is built with a
small torch cosine-kNN so the model carries no ``scikit-learn`` dependency.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models._external.marks import to_spatiotemporal


# --------------------------------------------------------------------------- #
# Vendored DCGRU cell (adapted from baselines/GTS/arch/gts_cell.py).
# --------------------------------------------------------------------------- #
class _LayerParams:
    """Lazily-allocated weight/bias store registered on its parent module."""

    def __init__(self, rnn_network: nn.Module, layer_type: str) -> None:
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
                f"{self._type}_weight_{shape}", nn_param
            )
        return self._params_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = nn.Parameter(torch.empty(length))
            nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter(
                f"{self._type}_biases_{length}", biases
            )
        return self._biases_dict[length]


class _DCGRUCell(nn.Module):
    """Diffusion-convolutional GRU cell over a (learned) adjacency."""

    def __init__(
        self,
        num_units: int,
        max_diffusion_step: int,
        num_nodes: int,
        nonlinearity: str = "tanh",
        use_gc_for_ru: bool = True,
    ) -> None:
        super().__init__()
        self._activation = torch.tanh if nonlinearity == "tanh" else torch.relu
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._use_gc_for_ru = use_gc_for_ru
        self._fc_params = _LayerParams(self, "fc")
        self._gconv_params = _LayerParams(self, "gconv")

    def _calculate_random_walk_matrix(self, adj_mx: torch.Tensor) -> torch.Tensor:
        adj_mx = adj_mx + torch.eye(int(adj_mx.shape[0]), device=adj_mx.device)
        d = torch.sum(adj_mx, 1)
        d_inv = 1.0 / d
        d_inv = torch.where(
            torch.isinf(d_inv), torch.zeros_like(d_inv), d_inv
        )
        d_mat_inv = torch.diag(d_inv)
        return torch.mm(d_mat_inv, adj_mx)

    def forward(self, inputs, hx, adj):
        adj_mx = self._calculate_random_walk_matrix(adj).t()
        output_size = 2 * self._num_units
        fn = self._gconv if self._use_gc_for_ru else self._fc
        value = torch.sigmoid(fn(inputs, adj_mx, hx, output_size, bias_start=1.0))
        value = torch.reshape(value, (-1, self._num_nodes, output_size))
        r, u = torch.split(value, self._num_units, dim=-1)
        r = torch.reshape(r, (-1, self._num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self._num_nodes * self._num_units))

        c = self._gconv(inputs, adj_mx, r * hx, self._num_units)
        if self._activation is not None:
            c = self._activation(c)
        return u * hx + (1.0 - u) * c

    @staticmethod
    def _concat(x, x_):
        return torch.cat([x, x_.unsqueeze(0)], dim=0)

    def _fc(self, inputs, adj_mx, state, output_size, bias_start=0.0):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size * self._num_nodes, -1))
        state = torch.reshape(state, (batch_size * self._num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=-1)
        input_size = inputs_and_state.shape[-1]
        weights = self._fc_params.get_weights((input_size, output_size))
        value = torch.sigmoid(torch.matmul(inputs_and_state, weights))
        biases = self._fc_params.get_biases(output_size, bias_start)
        value = value + biases
        return value

    def _gconv(self, inputs, adj_mx, state, output_size, bias_start=0.0):
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
            x1 = torch.mm(adj_mx, x0)
            x = self._concat(x, x1)
            for _ in range(2, self._max_diffusion_step + 1):
                x2 = 2 * torch.mm(adj_mx, x1) - x0
                x = self._concat(x, x2)
                x1, x0 = x2, x1
        num_matrices = self._max_diffusion_step + 1
        x = torch.reshape(
            x, shape=[num_matrices, self._num_nodes, input_size, batch_size]
        )
        x = x.permute(3, 1, 2, 0)
        x = torch.reshape(
            x, shape=[batch_size * self._num_nodes, input_size * num_matrices]
        )
        weights = self._gconv_params.get_weights(
            (input_size * num_matrices, output_size)
        ).to(x.device)
        x = torch.matmul(x, weights)
        biases = self._gconv_params.get_biases(output_size, bias_start).to(x.device)
        x = x + biases
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])


# --------------------------------------------------------------------------- #
# Vendored GTS arch helpers (adapted from baselines/GTS/arch/gts_arch.py).
# --------------------------------------------------------------------------- #
def _sample_gumbel(shape, eps=1e-20, device=None):
    u = torch.rand(shape, device=device)
    return -torch.log(-torch.log(u + eps) + eps)


def _gumbel_softmax_sample(logits, temperature, eps=1e-10):
    sample = _sample_gumbel(logits.size(), eps=eps, device=logits.device)
    y = logits + sample
    return F.softmax(y / temperature, dim=-1)


def _gumbel_softmax(logits, temperature, hard=False, eps=1e-10):
    y_soft = _gumbel_softmax_sample(logits, temperature=temperature, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        y_hard = torch.zeros(*shape, device=logits.device)
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        y = (y_hard - y_soft).detach() + y_soft
    else:
        y = y_soft
    return y


def _encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    return np.array(list(map(classes_dict.get, labels)), dtype=np.int32)


class _Seq2SeqAttrs:
    def __init__(self, **kw):
        self.max_diffusion_step = int(kw.get("max_diffusion_step", 2))
        self.cl_decay_steps = int(kw.get("cl_decay_steps", 1000))
        self.num_nodes = int(kw.get("num_nodes", 1))
        self.num_rnn_layers = int(kw.get("num_rnn_layers", 1))
        self.rnn_units = int(kw.get("rnn_units"))
        self.hidden_state_size = self.num_nodes * self.rnn_units


class _EncoderModel(nn.Module, _Seq2SeqAttrs):
    def __init__(self, **kw):
        nn.Module.__init__(self)
        _Seq2SeqAttrs.__init__(self, **kw)
        self.input_dim = int(kw.get("input_dim", 1))
        self.seq_len = int(kw.get("seq_len"))
        self.dcgru_layers = nn.ModuleList(
            [
                _DCGRUCell(self.rnn_units, self.max_diffusion_step, self.num_nodes)
                for _ in range(self.num_rnn_layers)
            ]
        )

    def forward(self, inputs, adj, hidden_state=None):
        batch_size, _ = inputs.size()
        if hidden_state is None:
            hidden_state = torch.zeros(
                (self.num_rnn_layers, batch_size, self.hidden_state_size),
                device=inputs.device,
            )
        hidden_states = []
        output = inputs
        for layer_num, layer in enumerate(self.dcgru_layers):
            next_hidden_state = layer(output, hidden_state[layer_num], adj)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state
        return output, torch.stack(hidden_states)


class _DecoderModel(nn.Module, _Seq2SeqAttrs):
    def __init__(self, **kw):
        nn.Module.__init__(self)
        _Seq2SeqAttrs.__init__(self, **kw)
        self.output_dim = int(kw.get("output_dim", 1))
        self.horizon = int(kw.get("horizon", 1))
        self.projection_layer = nn.Linear(self.rnn_units, self.output_dim)
        self.dcgru_layers = nn.ModuleList(
            [
                _DCGRUCell(self.rnn_units, self.max_diffusion_step, self.num_nodes)
                for _ in range(self.num_rnn_layers)
            ]
        )

    def forward(self, inputs, adj, hidden_state=None):
        hidden_states = []
        output = inputs
        for layer_num, layer in enumerate(self.dcgru_layers):
            next_hidden_state = layer(output, hidden_state[layer_num], adj)
            hidden_states.append(next_hidden_state)
            output = next_hidden_state
        projected = self.projection_layer(output.view(-1, self.rnn_units))
        output = projected.view(-1, self.num_nodes * self.output_dim)
        return output, torch.stack(hidden_states)


class _GTS(nn.Module, _Seq2SeqAttrs):
    """Upstream GTS network (graph structure learning + DCRNN recurrence)."""

    def __init__(self, node_feats: np.ndarray, k: int, temp: float, **kw):
        super().__init__()
        _Seq2SeqAttrs.__init__(self, **kw)
        self.encoder_model = _EncoderModel(**kw)
        self.decoder_model = _DecoderModel(**kw)
        self.cl_decay_steps = int(kw.get("cl_decay_steps", 1000))
        self.use_curriculum_learning = bool(kw.get("use_curriculum_learning", False))
        self.dim_fc = int(kw.get("dim_fc"))
        self.embedding_dim = int(kw.get("embedding_dim", 100))
        self.temp = float(temp)

        self.conv1 = nn.Conv1d(1, 8, 10, stride=1)
        self.conv2 = nn.Conv1d(8, 16, 10, stride=1)
        self.hidden_drop = nn.Dropout(0.2)
        self.fc = nn.Linear(self.dim_fc, self.embedding_dim)
        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(16)
        self.bn3 = nn.BatchNorm1d(self.embedding_dim)
        self.fc_out = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
        self.fc_cat = nn.Linear(self.embedding_dim, 2)

        off_diag = np.ones([self.num_nodes, self.num_nodes])
        rel_rec = np.array(_encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(_encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
        self.register_buffer("rel_rec", torch.FloatTensor(rel_rec))
        self.register_buffer("rel_send", torch.FloatTensor(rel_send))

        # node_feats: (feat_len, N) fixed feature series for graph learning.
        nf = torch.as_tensor(node_feats, dtype=torch.float32)
        self.register_buffer("node_feats", nf)

        # KL prior graph built with a dependency-free torch cosine-kNN.
        self.register_buffer("prior_adj", self._cosine_knn(nf, k))

    @staticmethod
    def _cosine_knn(node_feats: torch.Tensor, k: int) -> torch.Tensor:
        """Symmetric-free cosine kNN graph over columns (nodes) of node_feats.

        ``node_feats`` is ``(feat_len, N)``; we compare node columns. Returns an
        ``(N, N)`` binary adjacency with the ``k`` nearest neighbours per node.
        """
        feats = node_feats.t()  # (N, feat_len)
        norm = feats.norm(dim=1, keepdim=True).clamp(min=1e-8)
        feats = feats / norm
        sim = feats @ feats.t()  # (N, N) cosine similarity
        n = sim.shape[0]
        kk = max(1, min(k, n - 1))
        adj = torch.zeros_like(sim)
        # Exclude self by setting diagonal to -inf before top-k.
        sim = sim - torch.eye(n, device=sim.device) * 1e9
        idx = sim.topk(kk, dim=1).indices
        adj.scatter_(1, idx, 1.0)
        return adj

    def encoder(self, inputs, adj):
        encoder_hidden_state = None
        for t in range(self.encoder_model.seq_len):
            _, encoder_hidden_state = self.encoder_model(
                inputs[t], adj, encoder_hidden_state
            )
        return encoder_hidden_state

    def decoder(self, encoder_hidden_state, adj, labels=None, batches_seen=None):
        batch_size = encoder_hidden_state.size(1)
        go_symbol = torch.zeros(
            (batch_size, self.num_nodes * self.decoder_model.output_dim),
            device=encoder_hidden_state.device,
        )
        decoder_hidden_state = encoder_hidden_state
        decoder_input = go_symbol
        outputs = []
        for _ in range(self.decoder_model.horizon):
            decoder_output, decoder_hidden_state = self.decoder_model(
                decoder_input, adj, decoder_hidden_state
            )
            decoder_input = decoder_output
            outputs.append(decoder_output)
        return torch.stack(outputs)

    def forward(self, history_data, future_data=None, batch_seen=None, **kw):
        batch_size, length, num_nodes, channels = history_data.shape
        history_data = history_data.reshape(batch_size, length, num_nodes * channels)
        history_data = history_data.transpose(0, 1)  # (L, B, N*C)

        labels = None
        if future_data is not None:
            b2, l2, n2, c2 = future_data.shape
            future_data = future_data.reshape(b2, l2, n2 * c2).transpose(0, 1)
            labels = future_data

        inputs = history_data

        x = self.node_feats.transpose(1, 0).view(self.num_nodes, 1, -1)
        x = x.to(history_data.device)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = x.view(self.num_nodes, -1)
        x = self.fc(x)
        x = F.relu(x)
        x = self.bn3(x)

        receivers = torch.matmul(self.rel_rec.to(x.device), x)
        senders = torch.matmul(self.rel_send.to(x.device), x)
        x = torch.cat([senders, receivers], dim=1)
        x = torch.relu(self.fc_out(x))
        x = self.fc_cat(x)

        adj = _gumbel_softmax(x, temperature=self.temp, hard=True)
        adj = adj[:, 0].clone().reshape(self.num_nodes, -1)
        mask = torch.eye(self.num_nodes, self.num_nodes, device=adj.device).bool()
        adj = adj.masked_fill(mask, 0)

        encoder_hidden_state = self.encoder(inputs, adj)
        outputs = self.decoder(
            encoder_hidden_state, adj, labels, batches_seen=batch_seen
        )
        prediction = outputs.transpose(1, 0).unsqueeze(-1)  # (B, horizon, N*O, 1)
        prediction = prediction.reshape(
            batch_size, self.decoder_model.horizon, self.num_nodes, -1
        )
        return prediction


# --------------------------------------------------------------------------- #
# ModernTSF adapter.
# --------------------------------------------------------------------------- #
class Model(nn.Module):
    """Adapter wrapping the upstream GTS spatiotemporal graph model.

    Parameters
    ----------
    seq_len : int
        Input sequence length.
    pred_len : int
        Forecast horizon.
    num_nodes : int
        Number of spatial nodes ``N`` (injected from the dataset; falls back to
        ``enc_in``).
    adj_mx : np.ndarray or None
        Predefined ``(N, N)`` adjacency, injected by the runner. Used to seed the
        graph-learning feature series when present; GTS otherwise learns the graph
        from scratch.
    input_dim : int
        Number of input channels per node fed to the recurrence (value + the
        spatiotemporal covariate channels, i.e. ``1 + F``). Default 3.
    rnn_units : int
        Hidden size of the DCGRU cells.
    num_rnn_layers : int
        Number of stacked DCGRU layers.
    max_diffusion_step : int
        Diffusion / Chebyshev order of the graph convolution.
    embedding_dim : int
        Node embedding dimension for the graph-structure learner.
    node_feats_len : int
        Length of the internal per-node feature series used for graph learning.
        Must exceed 18 (two conv1d(kernel=10) layers consume 18 steps).
    k : int
        Neighbour count for the KL prior graph.
    temp : float
        Gumbel-Softmax temperature.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx=None,
        input_dim: int = 3,
        rnn_units: int = 16,
        num_rnn_layers: int = 1,
        max_diffusion_step: int = 2,
        embedding_dim: int = 16,
        node_feats_len: int = 40,
        k: int = 3,
        temp: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_nodes = num_nodes
        self.pred_len = pred_len

        node_feats = self._build_node_feats(num_nodes, node_feats_len, adj_mx)
        feat_len = node_feats.shape[0]
        # dim_fc: conv1(kernel 10) -> feat_len-9, conv2(kernel 10) -> feat_len-18,
        # flattened over the 16 output channels.
        dim_fc = (feat_len - 18) * 16

        self.net = _GTS(
            node_feats=node_feats,
            k=k,
            temp=temp,
            num_nodes=num_nodes,
            input_dim=input_dim,
            output_dim=1,
            seq_len=seq_len,
            horizon=pred_len,
            rnn_units=rnn_units,
            num_rnn_layers=num_rnn_layers,
            max_diffusion_step=max_diffusion_step,
            embedding_dim=embedding_dim,
            dim_fc=dim_fc,
        )

    @staticmethod
    def _build_node_feats(num_nodes, feat_len, adj_mx) -> np.ndarray:
        """Build a deterministic ``(feat_len, N)`` per-node feature series.

        GTS learns its graph from a per-node feature series. ModernTSF datasets
        expose only an adjacency, so we synthesise a smooth per-node series
        (phase-shifted sinusoids) and, when an adjacency is supplied, mix in a
        neighbour-averaged variant so structurally-related nodes look similar.
        """
        if feat_len <= 18:
            feat_len = 19
        t = np.arange(feat_len, dtype=np.float32)
        base = np.stack(
            [
                np.sin(2 * np.pi * t / feat_len + 2 * np.pi * i / max(1, num_nodes))
                for i in range(num_nodes)
            ],
            axis=1,
        )  # (feat_len, N)
        if adj_mx is not None:
            adj = np.asarray(adj_mx, dtype=np.float32)
            row_sum = adj.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1.0
            norm_adj = adj / row_sum
            base = 0.5 * base + 0.5 * (base @ norm_adj.T)
        return base.astype(np.float32)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forecast future node values.

        Parameters
        ----------
        x_enc : torch.Tensor
            Input values of shape ``(B, seq_len, N)``.
        x_mark_enc : torch.Tensor, optional
            Node covariates ``(B, seq_len, N, F)`` or raw stamps ``(B, seq_len, 6)``.
        x_dec, x_mark_dec, mask
            Unused by GTS (no teacher forcing in this adapter).

        Returns
        -------
        torch.Tensor
            Forecast of shape ``(B, pred_len, N)``.
        """
        history = to_spatiotemporal(x_enc, x_mark_enc)  # (B, L, N, 1+F)
        out = self.net(
            history, None, batch_seen=0, epoch=0, train=self.training
        )  # (B, pred_len, N, 1)
        if out.dim() == 4:
            return out[..., 0]
        return out.reshape(out.shape[0], self.pred_len, self.num_nodes)
