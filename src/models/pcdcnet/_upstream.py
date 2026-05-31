"""Upstream PCDCNet model ported from CauAir.

Verbatim logic with BaseModel replaced by nn.Module and explicit parameters.
All helper classes are bundled in this file.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper modules
# ---------------------------------------------------------------------------

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_ih, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_hh, a=math.sqrt(5))
        nn.init.zeros_(self.bias_ih)
        nn.init.zeros_(self.bias_hh)

    def forward(self, x, hx):
        gates = F.linear(x, self.weight_ih, self.bias_ih) + F.linear(hx, self.weight_hh, self.bias_hh)
        r, z, n = gates.chunk(3, dim=1)
        r = torch.sigmoid(r)
        z = torch.sigmoid(z)
        n = torch.tanh(n + r * hx)
        hy = (1 - z) * n + z * hx
        return hy


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, gso, num_layers=2,
                 dropout=0.1, drop_edge_p=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.gcn_layers = nn.ModuleList(
            [nn.Linear(in_features, out_features) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        self.drop_edge_p = drop_edge_p
        self.cached_adj = None
        self.register_buffer('adj', gso)

    def drop_edges(self, adj, drop_prob=0.1):
        if drop_prob <= 0:
            return adj
        mask = (torch.rand_like(adj) > drop_prob).float()
        return adj * mask

    def forward(self, x):
        b, n, f = x.shape
        h = x
        if self.cached_adj is None or self.training:
            adj = self.drop_edges(self.adj, self.drop_edge_p)
            self.cached_adj = adj.unsqueeze(0)
        adj = self.cached_adj.expand(b, -1, -1)
        for i in range(self.num_layers):
            h_new = torch.einsum("bmn,bnf->bmf", adj, h)
            h_new = self.gcn_layers[i](h_new)
            h_new = F.gelu(h_new)
            h_new = self.dropout(h_new)
            h = h + h_new
        return h


class PCDCNet(nn.Module):
    """PCDCNet: Physics-Constrained Deep Causal Network."""

    def __init__(self, adj_mx, node_num, input_dim, output_dim, seq_len, horizon,
                 d_model=64, num_layers=3, dropout=0.1):
        super().__init__()
        import numpy as np

        self.node_num = node_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.horizon = horizon
        self.hist_len = seq_len
        self.fut_len = horizon
        self.out_dim = output_dim
        self.in_dim = input_dim
        self.hid_size = d_model
        self.dropout_rate = dropout
        self.eps = 1e-6
        self.fmix_size = 4 * self.hid_size
        self.use_fmix = True
        self.use_spatial = True
        self.use_temporal = True
        self.use_adv = True

        gso = torch.tensor(adj_mx, dtype=torch.float32)
        self.embed = nn.Linear(self.in_dim, self.hid_size)
        self.readout = nn.Linear(self.hid_size, self.out_dim)
        self.out_norm = nn.RMSNorm([self.hid_size], eps=self.eps)

        if self.use_temporal:
            self.gru_cell = GRUCell(self.hid_size, self.hid_size)
            self.gru_norm = nn.RMSNorm([self.hid_size], eps=self.eps)
            self.hid_norm = nn.RMSNorm([self.hid_size], eps=self.eps)

        if self.use_spatial:
            self.gcn = GCNLayer(self.hid_size, self.hid_size, gso, dropout=self.dropout_rate)
            self.gnn_norm = nn.RMSNorm([self.hid_size], eps=self.eps)
            self.adv_lambda = 10
            self.adv_method = "mae"

        if self.use_fmix:
            self.feat_mix = nn.Sequential(
                nn.Linear(self.hid_size, self.fmix_size),
                nn.SiLU(),
                nn.Linear(self.fmix_size, self.hid_size),
                nn.Dropout(self.dropout_rate),
            )
            self.fmix_norm = nn.RMSNorm([self.hid_size], eps=self.eps)

        self.register_buffer('gso', gso)

    def forward(self, inputs, labels):
        """
        Parameters
        ----------
        inputs : (B, T_hist, N, F) history data
        labels : (B, T_fut, N, F-1) future covariates
        Returns
        -------
        (B, horizon, N, output_dim)
        """
        aqi_hist = inputs[..., :1]
        mete_hist = inputs[..., 1:]
        mete_fut = labels

        x_hist = mete_hist
        x_fut = mete_fut
        x_in = torch.cat([x_hist, x_fut], dim=1)  # (B, T_total, N, cov_dim)
        bs, ts, n, _ = x_in.shape

        if self.use_temporal:
            h_t = torch.zeros(bs * n, self.hid_size, device=x_in.device, dtype=x_in.dtype)

        all_preds = []
        aqi_t = aqi_hist[:, 0]

        for t in range(ts):
            if t < self.hist_len:
                aqi_t = aqi_hist[:, t]
            x_t = torch.cat((x_in[:, t], aqi_t), dim=-1)
            x_t = self.embed(x_t)

            if self.use_fmix:
                x_t = x_t + self.feat_mix(self.fmix_norm(x_t))

            if self.use_spatial:
                gcn_out = self.gcn(self.gnn_norm(x_t))
                x_t = x_t + gcn_out

            if self.use_temporal:
                h_t = self.gru_cell(
                    self.gru_norm(x_t.view(bs * n, -1)), self.hid_norm(h_t))
                x_t = x_t + h_t.view(bs, n, -1)

            x_delta = self.readout(self.out_norm(x_t))
            aqi_t = aqi_t + x_delta.view(bs, n, -1)
            all_preds.append(aqi_t)

        aqi_fut_hat = torch.stack(all_preds, dim=1)[:, self.hist_len:]
        return aqi_fut_hat
