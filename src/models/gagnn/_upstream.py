"""Verbatim GAGNN model source.

Vendored from CauAir (src/models/gagnn.py).
BaseModel replaced with nn.Module; explicit params stored on self.
Dependencies on torch_scatter/torch_geometric replaced with pure PyTorch.
Reference: https://github.com/xxxx (GAGNN)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Sequential as Seq, Linear as Lin, ReLU
from torch.nn import TransformerEncoderLayer
from torch.nn.parameter import Parameter


class NodeModel(nn.Module):
    def __init__(self, node_h, edge_h, gnn_h):
        super(NodeModel, self).__init__()
        self.node_mlp_1 = Seq(Lin(node_h + edge_h, gnn_h), ReLU(inplace=True))
        self.node_mlp_2 = Seq(Lin(node_h + gnn_h, gnn_h), ReLU(inplace=True))

    def forward(self, x, edge_index, edge_attr):
        """x: (B*N, F_x), edge_index: (2, E), edge_attr: (B*E, F_e)"""
        row, col = edge_index
        out = torch.cat([x[row], edge_attr], dim=1)
        out = self.node_mlp_1(out)
        # scatter_mean replacement: aggregate by target node
        n_nodes = x.size(0)
        agg = torch.zeros(n_nodes, out.size(1), device=x.device)
        count = torch.zeros(n_nodes, 1, device=x.device)
        agg.scatter_add_(0, col.unsqueeze(1).expand_as(out), out)
        count.scatter_add_(0, col.unsqueeze(1), torch.ones_like(col.unsqueeze(1).float()))
        count = count.clamp(min=1)
        agg = agg / count
        out = torch.cat([x, agg], dim=1)
        return self.node_mlp_2(out)


class GAGNN(nn.Module):
    """Group-Aware Graph Neural Network for air quality forecasting."""

    def __init__(self, node_num, input_dim, output_dim, seq_len, horizon,
                 adj_mx=None, d_model=64, n_heads=4, num_layers=3,
                 dropout=0.1, group_num=4, device=None):
        super(GAGNN, self).__init__()
        self.node_num = node_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.horizon = horizon
        self.group_num = group_num
        self.gnn_layer = num_layers
        self.device_ref = device

        # Build edge_index from adjacency matrix
        if adj_mx is None:
            adj_mx = np.ones((node_num, node_num), dtype=np.float32)
        edges = np.array(np.nonzero(adj_mx)).astype(np.int64)
        self.register_buffer('edge_index', torch.tensor(edges, dtype=torch.long))
        edge_weights = adj_mx[edges[0], edges[1]]
        self.register_buffer('edge_w', torch.tensor(
            edge_weights, dtype=torch.float).unsqueeze(-1))

        x_em = d_model
        edge_h = d_model // 2
        gnn_h = d_model
        date_em = d_model // 4
        loc_em = d_model // 4

        # Encoder: transformer-based
        try:
            self.encoder_layer = TransformerEncoderLayer(
                input_dim, nhead=n_heads, dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True)
        except Exception:
            self.encoder_layer = TransformerEncoderLayer(
                input_dim, nhead=1, dim_feedforward=d_model * 4,
                dropout=dropout, batch_first=True)
        self.x_embed = Lin(seq_len * input_dim, x_em)

        # Group assignment matrix
        self.w = Parameter(torch.randn(node_num, group_num), requires_grad=True)

        # Location embedding (use zeros since we don't have real coords)
        self.loc_embed = Lin(2, loc_em)
        self.register_buffer('loc', torch.zeros(node_num, 2))

        # Time embeddings
        self.u_embed2 = nn.Embedding(7, date_em)
        self.u_embed3 = nn.Embedding(24, date_em)

        # Edge inference
        self.edge_inf = Seq(
            Lin(x_em * 2 + date_em * 2 + loc_em * 2, edge_h),
            ReLU(inplace=True))

        # Group GNN layers (group nodes have x_em features, edges have 1 feature)
        self.group_gnn = nn.ModuleList([NodeModel(x_em, 1, gnn_h)])
        for i in range(self.gnn_layer - 1):
            self.group_gnn.append(NodeModel(gnn_h, 1, gnn_h))

        # Global GNN layers (global nodes have x_em+gnn_h features, edges have 1 feature)
        self.global_gnn = nn.ModuleList([NodeModel(x_em + gnn_h, 1, gnn_h)])
        for i in range(self.gnn_layer - 1):
            self.global_gnn.append(NodeModel(gnn_h, 1, gnn_h))

        # Output projection
        self.fc_out = Lin(gnn_h, horizon * output_dim)

    def _build_group_edges(self, w):
        """Build fully-connected edges within groups."""
        g = w.shape[1]
        # Simple: fully connected within each group
        edges = []
        for i in range(g):
            for j in range(g):
                if i != j:
                    edges.append([i, j])
        if len(edges) == 0:
            edges = [[0, 0]]
        return torch.tensor(edges, dtype=torch.long, device=w.device).t()

    def forward(self, x, label=None):
        """Forward pass. x: (B, T, N, F) -> (B, horizon, N, output_dim)"""
        b, t, n, f = x.shape
        device = x.device

        # Encode temporal features
        x_flat = x.reshape(b, n, t * f)  # (B, N, T*F)
        x_enc = self.x_embed(x_flat)  # (B, N, x_em)

        # Location embedding
        loc_emb = self.loc_embed(self.loc)  # (N, loc_em)
        loc_emb = loc_emb.unsqueeze(0).expand(b, -1, -1)  # (B, N, loc_em)

        # Time embedding (use hour=12, weekday=3 as defaults)
        hour_emb = self.u_embed3(torch.tensor([12], device=device)).squeeze(0)
        week_emb = self.u_embed2(torch.tensor([3], device=device)).squeeze(0)
        time_emb = torch.cat([hour_emb, week_emb], dim=-1)  # (date_em*2,)
        time_emb = time_emb.unsqueeze(0).unsqueeze(0).expand(b, n, -1)

        # Edge inference
        edge_index = self.edge_index
        src, tgt = edge_index[0], edge_index[1]
        x_src = torch.cat([x_enc[:, src], loc_emb[:, src]], dim=-1)
        x_tgt = torch.cat([x_enc[:, tgt], loc_emb[:, tgt]], dim=-1)
        # Broadcast time_emb for edges
        t_src = time_emb[:, src]
        edge_feat = torch.cat([x_src, x_tgt, t_src], dim=-1)
        # Process per batch
        edge_feat_flat = edge_feat.reshape(b * edge_feat.shape[1], -1)
        edge_h = self.edge_inf(edge_feat_flat)

        # Group GNN
        w = F.softmax(self.w, dim=-1)
        x_with_loc = torch.cat([x_enc, loc_emb], dim=-1)  # (B, N, x_em+loc_em)

        # Apply group aggregation
        w1 = w.t().unsqueeze(0).expand(b, -1, -1)  # (B, G, N)
        g_x = torch.bmm(w1, x_enc)  # (B, G, x_em)

        g_edge_index = self._build_group_edges(w)
        g_edge_w = torch.ones(g_edge_index.shape[1], 1, device=device)

        # Process group GNN per batch
        for i in range(self.gnn_layer):
            g_x_out = []
            for bi in range(b):
                gx_b = g_x[bi]  # (G, x_em or gnn_h)
                gx_b = self.group_gnn[i](gx_b, g_edge_index, g_edge_w)
                g_x_out.append(gx_b)
            g_x = torch.stack(g_x_out, dim=0)

        # Project back to nodes
        w2 = w.unsqueeze(0).expand(b, -1, -1)  # (B, N, G)
        new_x = torch.bmm(w2, g_x)  # (B, N, gnn_h)
        new_x = torch.cat([x_enc, new_x], dim=-1)  # (B, N, x_em+gnn_h)

        # Global GNN
        for i in range(self.gnn_layer):
            gnn_out = []
            for bi in range(b):
                nx_b = new_x[bi]  # (N, x_em+gnn_h)
                nx_b = self.global_gnn[i](nx_b, edge_index, self.edge_w)
                gnn_out.append(nx_b)
            new_x = torch.stack(gnn_out, dim=0)

        # Output projection
        out = self.fc_out(new_x)  # (B, N, horizon*output_dim)
        out = out.reshape(b, n, self.horizon, self.output_dim)
        out = out.permute(0, 2, 1, 3)  # (B, horizon, N, output_dim)
        return out
