"""Upstream AirDualODE model ported from CauAir.

Verbatim logic with BaseModel replaced by nn.Module and explicit parameters.
All helper classes are bundled in this file.

AirDualODE uses dual ODE systems (physics-informed + data-driven) with
knowledge fusion for air quality forecasting.
"""

import time as time_module
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.rnn import GRU

try:
    from torchdiffeq import odeint
except ImportError:
    odeint = None

import scipy.sparse as sp
from scipy.sparse import linalg


# ---------------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------------

def calculate_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)
def init_network_weights(net, std=0.1):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            nn.init.constant_(m.bias, val=0)


def split_last_dim(data):
    last_dim = data.size()[-1]
    last_dim = last_dim // 2
    return data[..., :last_dim], data[..., last_dim:]


# ---------------------------------------------------------------------------
# Chebyshev Graph Convolution
# ---------------------------------------------------------------------------

class ChebGraphConv(nn.Module):
    def __init__(self, c_in, c_out, Ks, gso):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.Ks = Ks
        self.register_buffer('gso', gso)
        self.weight = nn.Parameter(torch.FloatTensor(Ks, c_in, c_out))
        self.bias = nn.Parameter(torch.FloatTensor(c_out))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, edge_index=None, edge_attr=None):
        # x: (B, N, c_in)
        if self.Ks - 1 < 0:
            raise ValueError(f'Ks must be positive, got {self.Ks}')
        elif self.Ks - 1 == 0:
            x_list = [x]
        elif self.Ks - 1 == 1:
            x_0 = x
            x_1 = torch.einsum('hi,bij->bhj', self.gso, x)
            x_list = [x_0, x_1]
        else:
            x_0 = x
            x_1 = torch.einsum('hi,bij->bhj', self.gso, x)
            x_list = [x_0, x_1]
            for k in range(2, self.Ks):
                x_list.append(torch.einsum('hi,bij->bhj', 2 * self.gso, x_list[k - 1]) - x_list[k - 2])
        x = torch.stack(x_list, dim=1)
        cheb_graph_conv = torch.einsum('bkhi,kij->bhj', x, self.weight)
        cheb_graph_conv = torch.add(cheb_graph_conv, self.bias)
        return cheb_graph_conv

# ---------------------------------------------------------------------------
# ODE Functions
# ---------------------------------------------------------------------------

class Simple_Gated_Fusion(nn.Module):
    def __init__(self, num_nodes, var_dim):
        super().__init__()
        self.num_nodes = num_nodes
        self.var_dim = var_dim
        self.gated_fc = nn.Linear(2, 1)

    def forward(self, grad_diff, grad_adv):
        B = grad_diff.shape[0]
        grad_diff = grad_diff.reshape(B, self.num_nodes, self.var_dim)
        grad_adv = grad_adv.reshape(B, self.num_nodes, self.var_dim)
        concat = torch.cat((grad_diff, grad_adv), dim=-1)
        g = torch.sigmoid(self.gated_fc(concat))
        grad_diff_adv = g * grad_diff + (1 - g) * grad_adv
        return grad_diff_adv.reshape(B, self.num_nodes * self.var_dim)


class PhyODEFunc(nn.Module):
    """Physics-informed ODE function using Chebyshev graph convolution."""

    def __init__(self, gcn_hidden_dim, input_dim, num_nodes, gso, device,
                 num_layers=2, cheb_k=3):
        super().__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.gcn_hidden_dim = gcn_hidden_dim
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.nfe = 0
        self.diff_coeff = 0.1
        self.beta = nn.Parameter(torch.zeros(num_nodes * input_dim))
        self.diff_cheb_conv = nn.ModuleList()
        self.diff_cheb_conv.append(ChebGraphConv(input_dim, gcn_hidden_dim, cheb_k, gso))
        for _ in range(num_layers - 2):
            self.diff_cheb_conv.append(ChebGraphConv(gcn_hidden_dim, gcn_hidden_dim, cheb_k, gso))
        self.diff_cheb_conv.append(ChebGraphConv(gcn_hidden_dim, input_dim, cheb_k, gso))

    def forward(self, t, x):
        self.nfe += 1
        B = x.shape[0]
        h = x.reshape(B, self.num_nodes, self.input_dim)
        for i, conv in enumerate(self.diff_cheb_conv):
            h = conv(h)
            if i < len(self.diff_cheb_conv) - 1:
                h = torch.tanh(h)
        grad = self.diff_coeff * h.reshape(B, -1) - self.beta * x
        return grad

class UnkODEFunc(nn.Module):
    """Data-driven ODE function with attention."""

    def __init__(self, latent_dim, num_nodes, n_heads, device, adj_mask=None):
        super().__init__()
        self.nfe = 0
        self.latent_dim = latent_dim
        self.num_nodes = num_nodes
        if adj_mask is None:
            self.adj_mask = None
        else:
            self.adj_mask = torch.tensor(adj_mask, dtype=torch.int8,
                                         device=device) + torch.eye(num_nodes, device=device)
        self.fc = nn.Linear(latent_dim, latent_dim)
        self.spatial_att = nn.MultiheadAttention(latent_dim, num_heads=n_heads, batch_first=True)
        self.layer_norm_1 = nn.LayerNorm(latent_dim)
        self.layer_norm_2 = nn.LayerNorm(latent_dim)
        self.residual_1 = nn.Identity()
        self.residual_2 = nn.Identity()

    def forward(self, t, z):
        self.nfe += 1
        B = z.shape[0]
        z = z.reshape(B, self.num_nodes, self.latent_dim)
        z = self.residual_1(z) + self.spatial_att(z, z, z)[0]
        z = self.layer_norm_1(z)
        z = self.residual_2(z) + F.relu(self.fc(z))
        z = self.layer_norm_2(z)
        return z.reshape(B, self.num_nodes * self.latent_dim)


class DiffeqSolver:
    """Simple ODE solver wrapper."""

    def __init__(self, method, odeint_rtol=1e-5, odeint_atol=1e-5, adjoint=False):
        self.ode_method = method
        if adjoint:
            from torchdiffeq import odeint_adjoint as _odeint
        else:
            from torchdiffeq import odeint as _odeint
        self.odeint = _odeint
        self.rtol = odeint_rtol
        self.atol = odeint_atol

    def solve(self, odefunc, first_point, time_steps_to_pred):
        odefunc.nfe = 0
        pred_y = self.odeint(odefunc, first_point, time_steps_to_pred,
                             rtol=self.rtol, atol=self.atol, method=self.ode_method)
        return pred_y, (odefunc.nfe, 0)


# ---------------------------------------------------------------------------
# Encoders and Decoders
# ---------------------------------------------------------------------------

class Encoder_phy_z(nn.Module):
    def __init__(self, input_dim, latent_dim, num_layers, num_nodes):
        super().__init__()
        self.input_dim = input_dim
        self.phy_latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.gru_rnn = GRU(input_dim, latent_dim, num_layers=num_layers)

    def forward(self, X_p):
        T, B = X_p.size(0), X_p.size(1)
        X_p = X_p.reshape(T, B * self.num_nodes, self.input_dim)
        Z_p, _ = self.gru_rnn(X_p)
        Z_p = Z_p.reshape(T, B, self.num_nodes * self.phy_latent_dim)
        return Z_p

class Encoder_unk_z(nn.Module):
    def __init__(self, input_dim, latent_dim, num_nodes, rnn_dim, n_layers):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_nodes = num_nodes
        self.rnn_dim = rnn_dim
        self.gru_rnn = GRU(input_dim, rnn_dim, num_layers=n_layers)
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(rnn_dim, 50),
            nn.Tanh(),
            nn.Linear(50, latent_dim))
        init_network_weights(self.hiddens_to_z0)

    def forward(self, X):
        seq_len, batch_size = X.size(0), X.size(1)
        X = X.reshape(seq_len, batch_size * self.num_nodes, self.input_dim)
        outputs, _ = self.gru_rnn(X)
        last_output = outputs[-1]
        last_output = torch.reshape(last_output, (batch_size, self.num_nodes, self.rnn_dim))
        Z0 = self.hiddens_to_z0(last_output)
        Z0 = torch.reshape(Z0, (batch_size, self.num_nodes * self.latent_dim))
        return Z0


class Conv1d_Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim, num_nodes, k=1):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_nodes = num_nodes
        self.output_dim = output_dim
        padding = (k - 1) // 2
        self.decoder = nn.Sequential(
            nn.Conv1d(in_channels=latent_dim, out_channels=output_dim,
                      kernel_size=k, padding=padding, bias=True))

    def forward(self, z):
        T, B, _ = z.shape
        z = z.reshape(T, B * self.num_nodes, self.latent_dim)
        z = z.permute(1, 2, 0)
        z = self.decoder(z)
        z = z.permute(2, 0, 1)
        z = z.reshape(T, B, self.num_nodes * self.output_dim)
        return z


class GNN_Knowledge_Fusion(nn.Module):
    def __init__(self, num_nodes, phy_dim, unk_dim, output_dim, gso, hid_dim=64, num_layers=3):
        super().__init__()
        self.num_nodes = num_nodes
        self.phy_dim = phy_dim
        self.unk_dim = unk_dim
        self.concat_dim = phy_dim + unk_dim
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.residual = nn.Identity()
        self.fusion = nn.ModuleList()
        self.fusion.append(nn.Linear(self.concat_dim, hid_dim))
        for _ in range(num_layers - 2):
            self.fusion.append(nn.Linear(hid_dim, hid_dim))
        self.fusion.append(nn.Linear(hid_dim, output_dim))

    def forward(self, phy_hidden, unk_hidden):
        T, B, _ = phy_hidden.shape
        phy_hidden = phy_hidden.reshape(T * B, self.num_nodes, self.phy_dim)
        unk_hidden = unk_hidden.reshape(T * B, self.num_nodes, self.unk_dim)
        concat_hidden = torch.cat((phy_hidden, unk_hidden), dim=2)
        out = F.relu(self.fusion[0](concat_hidden))
        for layer in self.fusion[1:-1]:
            residual = self.residual(out)
            out = F.relu(layer(out)) + residual
        x = self.fusion[-1](out)
        return x.reshape(T, B, self.num_nodes * self.output_dim)

class AirDualODE(nn.Module):
    """AirDualODE: Dual ODE system for air quality forecasting.

    Combines physics-informed and data-driven dynamics with knowledge fusion.
    """

    def __init__(self, adj_mx, node_num, input_dim, output_dim, seq_len, horizon,
                 phy_latent_dim=16, unk_latent_dim=16, phy_rnn_dim=64,
                 unk_rnn_dim=64, gcn_hidden_dim=32, cheb_k=3,
                 n_heads=4, ode_method='euler', device=None):
        super().__init__()
        if device is None:
            device = torch.device('cpu')
        self.device = device
        self.node_num = node_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.horizon = horizon
        self.phy_latent_dim = phy_latent_dim
        self.unk_latent_dim = unk_latent_dim

        # Build GSO (graph shift operator) from adjacency
        L = calculate_scaled_laplacian(adj_mx)
        L_dense = L.toarray()
        gso = torch.tensor(L_dense, dtype=torch.float32)

        # Physics ODE
        self.phy_odefunc = PhyODEFunc(gcn_hidden_dim, output_dim, node_num, gso,
                                       device, num_layers=2, cheb_k=cheb_k)
        self.phy_solver = DiffeqSolver(method=ode_method, odeint_rtol=1e-4, odeint_atol=1e-5)
        self.RNN_encoder_pred = Encoder_phy_z(output_dim, phy_latent_dim, 1, node_num)

        # Unknown dynamics ODE
        self.encoder = Encoder_unk_z(input_dim, unk_latent_dim, node_num, unk_rnn_dim, 1)
        self.unk_odefunc = UnkODEFunc(unk_latent_dim, node_num, n_heads, device, adj_mx)
        self.unk_solver = DiffeqSolver(method=ode_method, odeint_rtol=1e-4, odeint_atol=1e-5)

        # Knowledge Fusion
        assert phy_latent_dim == unk_latent_dim
        fusion_output_dim = phy_latent_dim
        self.gatef_fusion = GNN_Knowledge_Fusion(
            node_num, phy_latent_dim, unk_latent_dim, fusion_output_dim, gso,
            hid_dim=64, num_layers=3)

        # Decoder
        self.decoder = Conv1d_Decoder(latent_dim=fusion_output_dim,
                                       output_dim=output_dim, num_nodes=node_num)

    def forward(self, inputs, labels=None):
        """
        Parameters
        ----------
        inputs : (B, T, N, F)
        Returns
        -------
        (B, horizon, N, output_dim)
        """
        batch_size = inputs.shape[0]
        seq_len = inputs.shape[1]
        inputs_t = inputs.transpose(0, 1)  # T x B x N x F
        inputs_t = inputs_t.reshape(seq_len, batch_size, self.node_num, self.input_dim)

        X = inputs_t[:, :, :, :self.output_dim].reshape(seq_len, batch_size, self.node_num * self.output_dim)
        last_X = X[-1]  # B x N*X_dim

        # Physics part
        time_steps = torch.arange(0, self.horizon + 1, dtype=torch.float32, device=inputs.device)
        time_steps = time_steps / len(time_steps)
        phy_y, _ = self.phy_solver.solve(self.phy_odefunc, last_X, time_steps)
        phy_y = phy_y[1:]  # T x B x N*X_dim
        phy_z = self.RNN_encoder_pred(phy_y.unsqueeze(-1).reshape(
            self.horizon, batch_size, self.node_num, self.output_dim).reshape(
            self.horizon, batch_size * self.node_num, self.output_dim).unsqueeze(0).squeeze(0).reshape(
            self.horizon, batch_size, self.node_num, self.output_dim))

        # Unknown dynamics part
        Z0 = self.encoder(inputs_t.reshape(seq_len, batch_size, self.node_num, self.input_dim))
        unk_z, _ = self.unk_solver.solve(self.unk_odefunc, Z0, time_steps)
        unk_z = unk_z[1:]  # T x B x N*unk_latent_dim

        # Fusion
        Z = self.gatef_fusion(phy_z, unk_z)
        pred_y = self.decoder(Z)

        # pred_y: T x B x N*output_dim -> B x T x N x output_dim
        pred_y = pred_y.transpose(0, 1).reshape(batch_size, self.horizon, self.node_num, self.output_dim)
        return pred_y
