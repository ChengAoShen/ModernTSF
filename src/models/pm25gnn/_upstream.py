"""Verbatim PM25_GNN model source.

Vendored from CauAir (src/models/PM25_GNN.py).
BaseModel replaced with nn.Module; explicit params stored on self.
torch_scatter replaced with pure PyTorch scatter operations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import Sequential, Linear, Sigmoid
from torch.nn import Parameter


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / np.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        x = x.view(-1, x.size(-1))
        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)
        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()
        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)
        resetgate = torch.sigmoid(i_r + h_r)
        inputgate = torch.sigmoid(i_i + h_i)
        newgate = torch.tanh(i_n + (resetgate * h_n))
        hy = newgate + inputgate * (hidden - newgate)
        return hy


class GraphGNN(nn.Module):
    def __init__(self, edge_index, edge_attr, in_dim, out_dim):
        super(GraphGNN, self).__init__()
        self.register_buffer(
            'edge_index', torch.LongTensor(edge_index))
        edge_attr_t = torch.tensor(np.float32(edge_attr))
        edge_attr_norm = (edge_attr_t - edge_attr_t.mean(dim=0)) / (
            edge_attr_t.std(dim=0) + 1e-8)
        self.register_buffer('edge_attr_norm', edge_attr_norm)
        self.w = Parameter(torch.rand([1]))
        self.b = Parameter(torch.rand([1]))
        e_h = 32
        e_out = 30
        n_out = out_dim
        self.edge_mlp = Sequential(
            Linear(in_dim * 2 + 2, e_h), Sigmoid(),
            Linear(e_h, e_out), Sigmoid())
        self.node_mlp = Sequential(
            Linear(e_out, n_out), Sigmoid())

    def forward(self, x, label=None):
        edge_index = self.edge_index
        edge_attr_norm = self.edge_attr_norm
        edge_src, edge_target = edge_index[..., 0], edge_index[..., 1]
        node_src = x[:, edge_src]
        node_target = x[:, edge_target]
        ea = edge_attr_norm[None, :, :].repeat(node_src.size(0), 1, 1)
        out = torch.cat([node_src, node_target, ea], dim=-1)
        out = self.edge_mlp(out)
        # scatter_add replacement
        n_nodes = x.size(1)
        b_size = x.size(0)
        out_add = torch.zeros(b_size, n_nodes, out.size(-1), device=x.device)
        out_add.scatter_add_(1, edge_target.unsqueeze(0).unsqueeze(-1).expand(
            b_size, -1, out.size(-1)), out)
        out_sub = torch.zeros(b_size, n_nodes, out.size(-1), device=x.device)
        out_sub.scatter_add_(1, edge_src.unsqueeze(0).unsqueeze(-1).expand(
            b_size, -1, out.size(-1)), out.neg())
        out = out_add + out_sub
        out = self.node_mlp(out)
        return out


class PM25_GNN(nn.Module):
    """PM2.5 GNN model for air quality forecasting."""

    def __init__(self, node_num, input_dim, output_dim, seq_len, horizon,
                 adj_mx=None, hid_dim=64):
        super(PM25_GNN, self).__init__()
        self.node_num = node_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.horizon = horizon
        self.hid_dim = hid_dim
        self.gnn_out = 13

        # Build edge_index and edge_attr from adjacency matrix
        if adj_mx is None:
            adj_mx = np.eye(node_num, dtype=np.float32)
        edges = np.array(np.nonzero(adj_mx))  # (2, E)
        edge_index = edges.T  # (E, 2)
        # Edge attributes: distance-like features from adj values
        edge_weights = adj_mx[edges[0], edges[1]]
        edge_attr = np.stack([edge_weights, np.ones_like(edge_weights)], axis=1)

        self.fc_in = nn.Linear(input_dim, hid_dim)
        self.graph_gnn = GraphGNN(edge_index, edge_attr, input_dim, self.gnn_out)
        self.gru_cell = GRUCell(input_dim + self.gnn_out, hid_dim)
        self.fc_out = nn.Linear(hid_dim, output_dim)

    def forward(self, x, label=None):
        """Forward pass. x: (B, T, N, F) -> (B, horizon, N, output_dim)"""
        b, t, n, f = x.shape
        device = x.device

        # Use last time step as initial input
        xn = x[:, -1, :, :self.output_dim]  # (B, N, output_dim)
        hn = torch.zeros(b * n, self.hid_dim, device=device)

        predictions = []
        for i in range(self.horizon):
            # Use features from input (repeat last step's features)
            feat = x[:, min(i, t - 1), :, :]  # (B, N, F)
            x_in = torch.cat([xn, feat[:, :, self.output_dim:]], dim=-1)

            # GNN pass
            xn_gnn = self.graph_gnn(x_in)
            x_combined = torch.cat([xn_gnn, x_in], dim=-1)

            # GRU step
            x_flat = x_combined.reshape(b * n, -1)
            hn = self.gru_cell(x_flat, hn)
            xn = hn.view(b, n, self.hid_dim)
            xn = self.fc_out(xn)  # (B, N, output_dim)
            predictions.append(xn)

        predictions = torch.stack(predictions, dim=1)  # (B, horizon, N, output_dim)
        return predictions
