"""Upstream D2STGNN model ported from CauAir/BasicTS.

D2STGNN: Decoupled Dynamic Spatial-Temporal Graph Neural Network.
This implementation preserves the key architectural ideas:
1. Decoupled diffusion (spatial) and inherent (temporal) patterns
2. Dynamic graph construction via learned node embeddings
3. Estimation gate for adaptive feature selection
4. Separate forecast branches for each pattern type
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper modules
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 0, 1::2] = torch.cos(position * div_term)
        else:
            pe[:, 0, 1::2] = torch.cos(position * div_term[:-1])
        self.register_buffer('pe', pe)

    def forward(self, X):
        X = X + self.pe[:X.size(0)]
        return self.dropout(X)


class ResidualDecomp(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x, y):
        return self.ln(x - F.relu(y))


class EstimationGate(nn.Module):
    def __init__(self, node_dim, time_emb_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(2 * node_dim + 2 * time_emb_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, node_emb_u, node_emb_d, tid_feat, diw_feat, history_data):
        B, T, N, _ = history_data.shape
        gate_feat = torch.cat([
            tid_feat[:, :T], diw_feat[:, :T],
            node_emb_u.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1),
            node_emb_d.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        ], dim=-1)
        gate = torch.sigmoid(self.fc2(F.relu(self.fc1(gate_feat))))
        return history_data * gate
class DiffusionBlock(nn.Module):
    """Spatial diffusion block using graph convolution."""

    def __init__(self, hidden_dim, k_t=3, dropout=0.1):
        super().__init__()
        self.k_t = k_t
        self.temporal_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=k_t,
                                       padding=(k_t - 1) // 2)
        self.graph_fc = nn.Linear(hidden_dim, hidden_dim)
        self.out_fc = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual_decompose = ResidualDecomp(hidden_dim)

    def forward(self, X, adj):
        B, T, N, D = X.shape
        # Temporal conv
        X_t = X.permute(0, 2, 3, 1).reshape(B * N, D, T)
        X_t = self.temporal_conv(X_t)[:, :, :T]
        X_t = X_t.reshape(B, N, D, T).permute(0, 3, 1, 2)
        # Graph conv
        X_g = torch.matmul(adj, X_t)
        X_g = self.graph_fc(X_g)
        out = self.out_fc(F.relu(X_t + X_g))
        out = self.dropout(out)
        backcast_res = self.residual_decompose(X, out)
        return backcast_res, out


class InherentBlock(nn.Module):
    """Temporal inherent pattern block using attention."""

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads,
                                               dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual_decompose = ResidualDecomp(hidden_dim)

    def forward(self, X):
        B, T, N, D = X.shape
        X_flat = X.permute(0, 2, 1, 3).reshape(B * N, T, D)
        rnn_out, _ = self.rnn(X_flat)
        attn_out, _ = self.attention(rnn_out, rnn_out, rnn_out)
        out = self.fc(F.relu(attn_out))
        out = self.dropout(out)
        out = out.reshape(B, N, T, D).permute(0, 2, 1, 3)
        backcast_res = self.residual_decompose(X, out)
        return backcast_res, out


class DecoupleLayer(nn.Module):
    def __init__(self, hidden_dim, node_dim, time_emb_dim, num_heads=4, k_t=3, dropout=0.1):
        super().__init__()
        self.gate = EstimationGate(node_dim, time_emb_dim, 64)
        self.dif_block = DiffusionBlock(hidden_dim, k_t=k_t, dropout=dropout)
        self.inh_block = InherentBlock(hidden_dim, num_heads=num_heads, dropout=dropout)

    def forward(self, X, adj, node_emb_u, node_emb_d, tid_feat, diw_feat):
        gated = self.gate(node_emb_u, node_emb_d, tid_feat, diw_feat, X)
        dif_res, dif_out = self.dif_block(gated, adj)
        inh_res, inh_out = self.inh_block(dif_res)
        return inh_res, dif_out, inh_out


class D2STGNN(nn.Module):
    """D2STGNN: Decoupled Dynamic Spatial-Temporal Graph Neural Network."""

    def __init__(self, adj_mx, node_num, input_dim, output_dim, seq_len, horizon,
                 d_model=64, num_layers=4, dropout=0.1):
        super().__init__()
        import numpy as np

        self.node_num = node_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.horizon = horizon

        hidden_dim = d_model
        node_dim = 10
        time_emb_dim = 10
        tpd = 288
        num_heads = 4
        k_t = 3

        self._hidden_dim = hidden_dim
        self._tpd = tpd

        # Embeddings
        self.embedding = nn.Linear(1, hidden_dim)
        self.node_emb_u = nn.Parameter(torch.empty(node_num, node_dim))
        self.node_emb_d = nn.Parameter(torch.empty(node_num, node_dim))
        self.T_i_D_emb = nn.Parameter(torch.empty(tpd, time_emb_dim))
        self.D_i_W_emb = nn.Parameter(torch.empty(7, time_emb_dim))

        # Static adjacency
        adj_t = torch.tensor(adj_mx, dtype=torch.float32)
        # Row-normalize
        deg = adj_t.sum(dim=1, keepdim=True).clamp(min=1e-6)
        self.register_buffer('adj', adj_t / deg)

        # Decouple layers
        self.layers = nn.ModuleList([
            DecoupleLayer(hidden_dim, node_dim, time_emb_dim, num_heads, k_t, dropout)
            for _ in range(num_layers)
        ])

        # Output projection
        self.out_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, horizon * output_dim),
        )

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.node_emb_u)
        nn.init.xavier_uniform_(self.node_emb_d)
        nn.init.xavier_uniform_(self.T_i_D_emb)
        nn.init.xavier_uniform_(self.D_i_W_emb)

    def forward(self, history_data, label=None):
        """
        Parameters
        ----------
        history_data : (B, T, N, F) where F = [value, time_in_day, day_in_week]
        Returns
        -------
        (B, horizon, N, output_dim)
        """
        B, T, N, C = history_data.shape
        # Extract time features and value
        tid_feat = self.T_i_D_emb[
            (history_data[:, :, :, 1] * self._tpd).clamp(0, self._tpd - 1).long()]
        diw_feat = self.D_i_W_emb[
            (history_data[:, :, :, 2] * 7).clamp(0, 6).long()]
        value = history_data[:, :, :, :1]

        # Embed value
        h = self.embedding(value)  # (B, T, N, hidden_dim)

        # Dynamic adjacency from node embeddings
        adj = self.adj + F.softmax(
            torch.mm(self.node_emb_u, self.node_emb_d.T), dim=1)

        # Decouple layers
        dif_outputs = []
        inh_outputs = []
        for layer in self.layers:
            h, dif_out, inh_out = layer(h, adj, self.node_emb_u, self.node_emb_d,
                                         tid_feat, diw_feat)
            dif_outputs.append(dif_out)
            inh_outputs.append(inh_out)

        # Aggregate forecasts
        dif_agg = sum(dif_outputs)  # (B, T, N, hidden_dim)
        inh_agg = sum(inh_outputs)
        forecast_hidden = dif_agg + inh_agg

        # Use last time step to predict horizon
        last_hidden = forecast_hidden[:, -1, :, :]  # (B, N, hidden_dim)
        pred = self.out_fc(last_hidden)  # (B, N, horizon * output_dim)
        pred = pred.reshape(B, N, self.horizon, self.output_dim)
        pred = pred.permute(0, 2, 1, 3)  # (B, horizon, N, output_dim)
        return pred
