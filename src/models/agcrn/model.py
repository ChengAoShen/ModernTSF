"""ModernTSF adapter for the AGCRN spatiotemporal graph forecasting model.

Vendored/adapted from https://github.com/GestaltCogTeam/BasicTS
(baselines/AGCRN), Apache-2.0.

AGCRN (Adaptive Graph Convolutional Recurrent Network, NeurIPS 2020) couples a
GRU with a node-adaptive graph convolution. It *learns* its adjacency from
trainable node embeddings (``supports = softmax(relu(E E^T))``), so no external
adjacency is strictly required; the ``adj_mx`` argument is accepted to honour
the ModernTSF graph-model contract but is unused by the upstream arch.

The upstream backbone keeps the BasicTS signature ``forward(history_data,
future_data, batch_seen, epoch, train, **kwargs)`` with ``history_data`` shaped
``(B, L, N, C)`` and returns ``(B, pred_len, N, output_dim)``.

This adapter:
  * converts ModernTSF's ``(x_enc, x_mark_enc)`` into ``(B, L, N, input_dim)``
    via :func:`models._external.marks.to_spatiotemporal` (channel 0 the value,
    then calendar ``[time_in_day, day_in_week]`` / node covariates),
  * drives the upstream module with the BasicTS signature (dummy
    ``future_data=None``),
  * squeezes the output channel back to ``(B, pred_len, N)``.

All hardcoded CUDA calls are removed; internally created tensors (the identity
in the Chebyshev support, the GRU initial hidden state) follow the input
tensor's device.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models._external.marks import to_spatiotemporal


# --------------------------------------------------------------------------- #
# Vendored AGCRN layers (adapted from BasicTS baselines/AGCRN/arch).
# --------------------------------------------------------------------------- #
class AVWGCN(nn.Module):
    """Adaptive node-specific graph convolution (Chebyshev over learned adj)."""

    def __init__(self, dim_in: int, dim_out: int, cheb_k: int, embed_dim: int) -> None:
        super().__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(
            torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out)
        )
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))

    def forward(self, x: torch.Tensor, node_embeddings: torch.Tensor) -> torch.Tensor:
        # x: (B, N, C); node_embeddings: (N, D) -> supports: (N, N)
        node_num = node_embeddings.shape[0]
        supports = F.softmax(
            F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1
        )
        # Identity follows the support tensor's device (no hardcoded CUDA).
        support_set = [torch.eye(node_num, device=supports.device), supports]
        # default cheb_k = 3
        for _ in range(2, self.cheb_k):
            support_set.append(
                torch.matmul(2 * supports, support_set[-1]) - support_set[-2]
            )
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum(
            "nd,dkio->nkio", node_embeddings, self.weights_pool
        )  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)  # N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)  # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum("bnki,nkio->bno", x_g, weights) + bias  # B, N, dim_out
        return x_gconv


class AGCRNCell(nn.Module):
    """GRU cell whose gates are adaptive graph convolutions."""

    def __init__(
        self, node_num: int, dim_in: int, dim_out: int, cheb_k: int, embed_dim: int
    ) -> None:
        super().__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AVWGCN(dim_in + self.hidden_dim, 2 * dim_out, cheb_k, embed_dim)
        self.update = AVWGCN(dim_in + self.hidden_dim, dim_out, cheb_k, embed_dim)

    def forward(
        self, x: torch.Tensor, state: torch.Tensor, node_embeddings: torch.Tensor
    ) -> torch.Tensor:
        # x: (B, N, input_dim); state: (B, N, hidden_dim)
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)


class AVWDCRNN(nn.Module):
    """Stack of adaptive-graph GRU cells (the AGCRN encoder)."""

    def __init__(
        self,
        node_num: int,
        dim_in: int,
        dim_out: int,
        cheb_k: int,
        embed_dim: int,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        assert num_layers >= 1, "At least one DCRNN layer in the Encoder."
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(
                AGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim)
            )

    def forward(
        self, x: torch.Tensor, init_state: torch.Tensor, node_embeddings: torch.Tensor
    ):
        # x: (B, T, N, D); init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](
                    current_inputs[:, t, :, :], state, node_embeddings
                )
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)  # (num_layers, B, N, hidden_dim)


class AGCRN(nn.Module):
    """Upstream AGCRN backbone.

    Paper: Adaptive Graph Convolutional Recurrent Network for Traffic
    Forecasting (NeurIPS 2020). Keeps the BasicTS ``forward`` signature.
    """

    def __init__(
        self,
        num_nodes: int,
        input_dim: int,
        rnn_units: int,
        output_dim: int,
        horizon: int,
        num_layers: int,
        embed_dim: int,
        cheb_k: int,
    ) -> None:
        super().__init__()
        self.num_node = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = rnn_units
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers

        self.node_embeddings = nn.Parameter(
            torch.randn(self.num_node, embed_dim), requires_grad=True
        )
        self.encoder = AVWDCRNN(
            num_nodes, input_dim, rnn_units, cheb_k, embed_dim, num_layers
        )
        # CNN-based predictor.
        self.end_conv = nn.Conv2d(
            1, horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True
        )
        self.init_param()

    def init_param(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor,
        batch_seen: int,
        epoch: int,
        train: bool,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        history_data : torch.Tensor
            Inputs with shape ``(B, L, N, C)``.

        Returns
        -------
        torch.Tensor
            Outputs with shape ``(B, horizon, N, output_dim)``.
        """
        init_state = self.encoder.init_hidden(history_data.shape[0])
        output, _ = self.encoder(
            history_data, init_state, self.node_embeddings
        )  # (B, T, N, hidden)
        output = output[:, -1:, :, :]  # (B, 1, N, hidden)

        output = self.end_conv(output)  # (B, T*C, N, 1)
        output = output.squeeze(-1).reshape(
            -1, self.horizon, self.output_dim, self.num_node
        )
        output = output.permute(0, 1, 3, 2)  # (B, T, N, C)
        return output


# --------------------------------------------------------------------------- #
# ModernTSF adapter.
# --------------------------------------------------------------------------- #
class Model(nn.Module):
    """ModernTSF adapter wrapping the upstream AGCRN backbone.

    Parameters
    ----------
    seq_len : int
        Input sequence length (``L``).
    pred_len : int
        Forecast horizon.
    num_nodes : int
        Number of spatial nodes ``N`` (= ``enc_in``). Injected from the dataset.
    adj_mx : np.ndarray | None
        ``(N, N)`` adjacency matrix injected by the runner. AGCRN learns its own
        adjacency from node embeddings, so this is accepted for the contract but
        not used by the backbone.
    input_dim : int
        Number of input channels fed to AGCRN (value + calendar covariates).
    rnn_units : int
        GRU hidden width.
    embed_dim : int
        Node-embedding dimension for the adaptive graph.
    num_layers : int
        Number of stacked adaptive-graph GRU layers.
    cheb_k : int
        Chebyshev order of the adaptive graph convolution.
    output_dim : int
        Per-node output channels (1 for univariate forecasting).
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx=None,
        input_dim: int = 3,
        rnn_units: int = 32,
        embed_dim: int = 8,
        num_layers: int = 1,
        cheb_k: int = 2,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.output_dim = output_dim

        # AGCRN learns its adjacency from node embeddings; ``adj_mx`` is accepted
        # for the ModernTSF graph contract but intentionally unused. Keep a
        # device-following buffer if one is supplied so it travels with the
        # model (purely for inspection / future use).
        if adj_mx is not None:
            adj_np = np.asarray(adj_mx, dtype=np.float32)
            self.register_buffer("adj_mx", torch.from_numpy(adj_np), persistent=False)
        else:
            self.adj_mx = None

        self.net = AGCRN(
            num_nodes=num_nodes,
            input_dim=input_dim,
            rnn_units=rnn_units,
            output_dim=output_dim,
            horizon=pred_len,
            num_layers=num_layers,
            embed_dim=embed_dim,
            cheb_k=cheb_k,
        )

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
            Node covariates ``(B, seq_len, N, F)`` or raw calendar stamps
            ``(B, seq_len, 6)``.

        Returns
        -------
        torch.Tensor
            Forecast of shape ``(B, pred_len, N)``.
        """
        # (B, L, N, 1 + F); channel 0 value, then calendar [tod, dow] / covariates.
        history = to_spatiotemporal(x_enc, x_mark_enc)
        # Keep / pad to exactly ``input_dim`` channels.
        history = history[..., : self.input_dim]
        if history.shape[-1] < self.input_dim:
            pad = history.new_zeros(
                (*history.shape[:-1], self.input_dim - history.shape[-1])
            )
            history = torch.cat([history, pad], dim=-1)

        out = self.net(history, None, batch_seen=0, epoch=0, train=self.training)
        # out is (B, pred_len, N, output_dim).
        if out.dim() == 4:
            return out[..., 0]
        return out.reshape(out.shape[0], self.pred_len, self.num_nodes)
