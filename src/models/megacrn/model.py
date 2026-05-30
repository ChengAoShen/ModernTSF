"""ModernTSF adapter for the MegaCRN spatiotemporal forecasting model.

Vendored/adapted from https://github.com/GestaltCogTeam/BasicTS
(baselines/MegaCRN), Apache-2.0.

MegaCRN (AAAI 2023, "Spatio-Temporal Meta-Graph Learning for Traffic
Forecasting") is a memory-augmented adaptive graph convolutional recurrent
network. It learns its graph entirely from a memory bank (node embeddings
factorised through the memory), so a predefined adjacency matrix is *optional*;
when one is supplied we append it (row-normalised) as an extra support to the
two learned graphs.

The upstream ``forward(history_data, future_data, batch_seen, epoch, train)``
consumes ``history_data`` of shape ``(B, L, N, C)`` (channel 0 = value) and a
``future_data`` block whose channel 0 carries the target labels (for curriculum
learning) and channel 1 a known future covariate ``y_cov``. The adapter builds
both tensors from ModernTSF's ``(x_enc, marks)`` and returns ``(B, pred_len,
N)``.

All hardcoded CUDA placement from upstream has been removed; every internally
created tensor (identity, hidden state, decoder ``go``) follows the input
tensor's device.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models._external.marks import to_spatiotemporal


# --------------------------------------------------------------------------- #
# Vendored upstream architecture (adapted: device-agnostic, no cuda).
# --------------------------------------------------------------------------- #
class AGCN(nn.Module):
    """Adaptive graph convolution with Chebyshev support expansion."""

    def __init__(self, dim_in, dim_out, cheb_k, num_support=2):
        super().__init__()
        self.cheb_k = cheb_k
        # ``num_support`` is the number of base supports passed to ``forward``
        # (2 learned graphs + any predefined adjacency); fixed at construction
        # so the weight matrix is sized correctly for the optimizer.
        self.weights = nn.Parameter(
            torch.empty(num_support * cheb_k * dim_in, dim_out)
        )
        self.bias = nn.Parameter(torch.zeros(dim_out))
        nn.init.xavier_normal_(self.weights)

    def forward(self, x, supports):
        support_set = []
        for support in supports:
            support_ks = [torch.eye(support.shape[0], device=support.device), support]
            for k in range(2, self.cheb_k):
                support_ks.append(
                    torch.matmul(2 * support, support_ks[-1]) - support_ks[-2]
                )
            support_set.extend(support_ks)
        x_g = []
        for support in support_set:
            x_g.append(torch.einsum("nm,bmc->bnc", support, x))
        x_g = torch.cat(x_g, dim=-1)  # B, N, num_support * cheb_k * dim_in
        x_gconv = torch.einsum("bni,io->bno", x_g, self.weights) + self.bias
        return x_gconv


class AGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_support=2):
        super().__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = AGCN(dim_in + self.hidden_dim, 2 * dim_out, cheb_k, num_support)
        self.update = AGCN(dim_in + self.hidden_dim, dim_out, cheb_k, num_support)

    def forward(self, x, state, supports):
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, supports))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z * state), dim=-1)
        hc = torch.tanh(self.update(candidate, supports))
        h = r * state + (1 - r) * hc
        return h

    def init_hidden_state(self, batch_size, device):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim, device=device)


class ADCRNN_Encoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers, num_support=2):
        super().__init__()
        assert num_layers >= 1, "At least one DCRNN layer in the Encoder."
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, num_support))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(
                AGCRNCell(node_num, dim_out, dim_out, cheb_k, num_support)
            )

    def forward(self, x, init_state, supports):
        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, supports)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size, device):
        return [
            self.dcrnn_cells[i].init_hidden_state(batch_size, device)
            for i in range(self.num_layers)
        ]


class ADCRNN_Decoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, num_layers, num_support=2):
        super().__init__()
        assert num_layers >= 1, "At least one DCRNN layer in the Decoder."
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(AGCRNCell(node_num, dim_in, dim_out, cheb_k, num_support))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(
                AGCRNCell(node_num, dim_out, dim_out, cheb_k, num_support)
            )

    def forward(self, xt, init_state, supports):
        assert xt.shape[1] == self.node_num and xt.shape[2] == self.input_dim
        current_inputs = xt
        output_hidden = []
        for i in range(self.num_layers):
            state = self.dcrnn_cells[i](current_inputs, init_state[i], supports)
            output_hidden.append(state)
            current_inputs = state
        return current_inputs, output_hidden


class MegaCRN(nn.Module):
    """Memory-augmented adaptive graph convolutional recurrent network.

    Paper: Spatio-Temporal Meta-Graph Learning for Traffic Forecasting
    Link: https://arxiv.org/abs/2212.05989
    Official Code: https://github.com/deepkashiwa20/MegaCRN
    Venue: AAAI 2023
    """

    def __init__(
        self,
        num_nodes,
        input_dim,
        output_dim,
        horizon,
        rnn_units,
        num_layers=1,
        cheb_k=3,
        ycov_dim=1,
        mem_num=20,
        mem_dim=64,
        cl_decay_steps=2000,
        use_curriculum_learning=True,
        predefined_supports=0,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.rnn_units = rnn_units
        self.output_dim = output_dim
        self.horizon = horizon
        self.num_layers = num_layers
        self.cheb_k = cheb_k
        self.ycov_dim = ycov_dim
        self.cl_decay_steps = cl_decay_steps
        self.use_curriculum_learning = use_curriculum_learning
        self.predefined_supports = predefined_supports

        self.mem_num = mem_num
        self.mem_dim = mem_dim
        self.memory = self.construct_memory()

        # 2 learned graphs plus any predefined adjacency support(s).
        num_support = 2 + predefined_supports

        self.encoder = ADCRNN_Encoder(
            self.num_nodes,
            self.input_dim,
            self.rnn_units,
            self.cheb_k,
            self.num_layers,
            num_support,
        )

        self.decoder_dim = self.rnn_units + self.mem_dim
        self.decoder = ADCRNN_Decoder(
            self.num_nodes,
            self.output_dim + self.ycov_dim,
            self.decoder_dim,
            self.cheb_k,
            self.num_layers,
            num_support,
        )

        self.proj = nn.Sequential(nn.Linear(self.decoder_dim, self.output_dim, bias=True))

    def compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
            self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps)
        )

    def construct_memory(self):
        memory_dict = nn.ParameterDict()
        memory_dict["Memory"] = nn.Parameter(
            torch.randn(self.mem_num, self.mem_dim), requires_grad=True
        )
        memory_dict["Wq"] = nn.Parameter(
            torch.randn(self.rnn_units, self.mem_dim), requires_grad=True
        )
        memory_dict["We1"] = nn.Parameter(
            torch.randn(self.num_nodes, self.mem_num), requires_grad=True
        )
        memory_dict["We2"] = nn.Parameter(
            torch.randn(self.num_nodes, self.mem_num), requires_grad=True
        )
        for param in memory_dict.values():
            nn.init.xavier_normal_(param)
        return memory_dict

    def query_memory(self, h_t: torch.Tensor):
        query = torch.matmul(h_t, self.memory["Wq"])
        att_score = torch.softmax(
            torch.matmul(query, self.memory["Memory"].t()), dim=-1
        )
        value = torch.matmul(att_score, self.memory["Memory"])
        _, ind = torch.topk(att_score, k=2, dim=-1)
        pos = self.memory["Memory"][ind[:, :, 0]]
        neg = self.memory["Memory"][ind[:, :, 1]]
        return value, query, pos, neg

    def forward(
        self,
        history_data,
        future_data,
        batch_seen=None,
        epoch=None,
        train=False,
        extra_supports=None,
        **kwargs,
    ):
        x = history_data[..., [0]]
        y_cov = future_data[..., [1]]
        labels = future_data[..., [0]]

        node_embeddings1 = torch.matmul(self.memory["We1"], self.memory["Memory"])
        node_embeddings2 = torch.matmul(self.memory["We2"], self.memory["Memory"])
        g1 = F.softmax(F.relu(torch.mm(node_embeddings1, node_embeddings2.T)), dim=-1)
        g2 = F.softmax(F.relu(torch.mm(node_embeddings2, node_embeddings1.T)), dim=-1)
        supports = [g1, g2]
        if extra_supports is not None:
            supports = supports + list(extra_supports)

        init_state = self.encoder.init_hidden(x.shape[0], x.device)
        h_en, state_en = self.encoder(x, init_state, supports)
        h_t = h_en[:, -1, :, :]

        h_att, query, pos, neg = self.query_memory(h_t)
        h_t = torch.cat([h_t, h_att], dim=-1)

        ht_list = [h_t] * self.num_layers
        go = torch.zeros(
            (x.shape[0], self.num_nodes, self.output_dim), device=x.device
        )
        out = []
        for t in range(self.horizon):
            h_de, ht_list = self.decoder(
                torch.cat([go, y_cov[:, t, ...]], dim=-1), ht_list, supports
            )
            go = self.proj(h_de)
            out.append(go)
            if train and self.use_curriculum_learning:
                c = np.random.uniform(0, 1)
                if c < self.compute_sampling_threshold(batch_seen or 0):
                    go = labels[:, t, ...]
        output = torch.stack(out, dim=1)
        return {"prediction": output, "query": query, "pos": pos, "neg": neg}


# --------------------------------------------------------------------------- #
# ModernTSF adapter.
# --------------------------------------------------------------------------- #
class Model(nn.Module):
    """Adapter wrapping the upstream MegaCRN model.

    Parameters
    ----------
    seq_len : int
        Input sequence length.
    pred_len : int
        Forecast horizon.
    num_nodes : int
        Number of spatial nodes ``N``.
    adj_mx : np.ndarray, optional
        Predefined ``(N, N)`` adjacency. Optional — MegaCRN learns its graph
        from the memory bank; when provided it is added (row-normalised) as an
        extra support.
    input_dim : int
        Number of channels in the reconstructed ``(B, L, N, input_dim)``
        spatiotemporal tensor (value + calendar covariates). Only channel 0 is
        consumed by the encoder, and channel 1 is reused as the known future
        covariate ``y_cov`` on the decoder side.
    rnn_units : int
        GRU hidden dimension.
    num_layers : int
        Number of recurrent layers.
    cheb_k : int
        Chebyshev polynomial order for the graph convolution.
    mem_num : int
        Number of memory slots.
    mem_dim : int
        Memory slot dimension.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx=None,
        input_dim: int = 3,
        rnn_units: int = 32,
        num_layers: int = 1,
        cheb_k: int = 3,
        mem_num: int = 8,
        mem_dim: int = 16,
        use_curriculum_learning: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.input_dim = input_dim

        has_adj = adj_mx is not None
        self.net = MegaCRN(
            num_nodes=num_nodes,
            input_dim=1,  # encoder consumes the value channel only
            output_dim=1,
            horizon=pred_len,
            rnn_units=rnn_units,
            num_layers=num_layers,
            cheb_k=cheb_k,
            ycov_dim=1,
            mem_num=mem_num,
            mem_dim=mem_dim,
            use_curriculum_learning=use_curriculum_learning,
            predefined_supports=1 if has_adj else 0,
        )

        # Register a row-normalised predefined support as a buffer (or None).
        if adj_mx is not None:
            adj = np.asarray(adj_mx, dtype=np.float32)
            row_sum = adj.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1.0
            adj_norm = adj / row_sum
            self.register_buffer("predefined_adj", torch.from_numpy(adj_norm))
        else:
            self.predefined_adj = None

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forecast future values.

        Parameters
        ----------
        x_enc : torch.Tensor
            Input values of shape ``(B, seq_len, N)``.
        x_mark_enc : torch.Tensor, optional
            Input marks: raw stamps ``(B, seq_len, 6)`` or node-structured
            covariates ``(B, seq_len, N, F)``.
        x_dec, x_mark_dec, mask
            Decoder inputs; ``x_mark_dec`` supplies the known future covariates
            when available.

        Returns
        -------
        torch.Tensor
            Forecast of shape ``(B, pred_len, N)``.
        """
        # History: (B, L, N, 1+F); channel 0 = value, then calendar [tod, dow].
        history = to_spatiotemporal(x_enc, x_mark_enc)

        b, n = history.shape[0], history.shape[2]
        # Future block (B, pred_len, N, >=2): channel 0 = labels (unused at
        # inference / zero), channel 1 = known future covariate y_cov.
        future = history.new_zeros((b, self.pred_len, n, 2))
        if x_mark_dec is not None:
            fut_st = to_spatiotemporal(
                x_enc.new_zeros((b, x_mark_dec.shape[1], n)), x_mark_dec
            )
            # take last pred_len future steps, channel 1 as y_cov
            fut_cov = fut_st[:, -self.pred_len :, :, 1:2]
            if fut_cov.shape[1] == self.pred_len:
                future[..., 1:2] = fut_cov
        if x_dec is not None:
            # decoder value channel (label_len + pred_len, N) -> last pred_len
            dec_val = x_dec[:, -self.pred_len :, :]
            if dec_val.shape[1] == self.pred_len:
                future[..., 0] = dec_val

        extra = None
        if self.predefined_adj is not None:
            extra = [self.predefined_adj.to(history.device)]

        out = self.net(
            history,
            future,
            batch_seen=0,
            epoch=0,
            train=self.training,
            extra_supports=extra,
        )
        pred = out["prediction"]  # (B, pred_len, N, 1)
        if pred.dim() == 4:
            return pred[..., 0]
        return pred.reshape(b, self.pred_len, n)
