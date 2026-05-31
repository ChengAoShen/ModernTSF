"""Upstream AirPhyNet model ported from CauAir.

Verbatim logic with BaseModel replaced by nn.Module and explicit parameters.
All helper classes are bundled in this file.
"""

import time as time_module
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
def sample_standard_gaussian(mu, sigma):
    device = mu.device
    d = torch.distributions.normal.Normal(torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))
    r = d.sample(mu.size()).squeeze(-1)
    return r * sigma.float() + mu.float()


def split_last_dim(data):
    last_dim = data.size()[-1]
    last_dim = last_dim // 2
    return data[..., :last_dim], data[..., last_dim:]


def init_network_weights(net, std=0.1):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0, std=std)
            nn.init.constant_(m.bias, val=0)


# ---------------------------------------------------------------------------
# Model components
# ---------------------------------------------------------------------------

class LayerParams:
    def __init__(self, rnn_network, layer_type, device=None):
        self._type = layer_type
        self._weights_dict = {}
        self._biases_dict = {}
        self._rnn_network = rnn_network
        self._device = device

    def get_weights(self, shape):
        if shape not in self._weights_dict:
            weights = nn.Parameter(torch.empty(*shape, device=self._device))
            nn.init.xavier_normal_(weights)
            self._weights_dict[shape] = weights
            self._rnn_network.register_parameter(
                '{}_weights_{}'.format(self._type, str(shape)), weights)
        return self._weights_dict[shape]

    def get_biases(self, length, bias_start=0.0):
        if length not in self._biases_dict:
            biases = nn.Parameter(torch.empty(length, device=self._device))
            nn.init.constant_(biases, bias_start)
            self._biases_dict[length] = biases
            self._rnn_network.register_parameter(
                '{}_biases_{}'.format(self._type, str(length)), biases)
        return self._biases_dict[length]


class LinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 1)

    def forward(self, x):
        return self.fc(x)

class GatedFusionModel(nn.Module):
    def __init__(self, num_nodes, latent_dim):
        super().__init__()
        self._num_nodes = num_nodes
        self._latent_dim = latent_dim
        self.hid_dim = self._num_nodes * self._latent_dim
        self.fc = nn.Linear(self.hid_dim, self.hid_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, grad_diff, grad_adv):
        X_diff = self.fc(grad_diff)
        X_adv = self.fc(grad_adv)
        z = self.sigmoid(torch.add(X_diff, X_adv))
        H = torch.add((z * X_diff), ((1 - z) * X_adv))
        return H


class ODEFunc(nn.Module):
    def __init__(self, num_units, latent_dim, adj_mx, gcn_step, num_nodes,
                 gen_layers=1, nonlinearity='tanh', filter_type="diff", device=None):
        super().__init__()
        self._activation = torch.tanh if nonlinearity == 'tanh' else torch.relu
        self._num_nodes = num_nodes
        self._num_units = num_units
        self._latent_dim = latent_dim
        self._gen_layers = gen_layers
        self.nfe = 0
        self._filter_type = filter_type
        self._device = device
        self._gcn_step = gcn_step
        self._gconv_params = LayerParams(self, 'gconv', device=device)
        self._supports = []
        supports = []
        supports.append(calculate_scaled_laplacian(adj_mx))
        for support in supports:
            self._supports.append(self._build_sparse_matrix(support, device))

    @staticmethod
    def _build_sparse_matrix(L, device):
        L = L.tocoo()
        indices = np.column_stack((L.row, L.col))
        indices = torch.LongTensor(indices).t().to(device)
        data = torch.FloatTensor(L.data).to(device)
        return torch.sparse_coo_tensor(indices, data, torch.Size(L.shape))

    @staticmethod
    def _concat(x, x_):
        x_ = x_.unsqueeze(0)
        return torch.cat([x, x_], dim=0)

    def forward(self, t, inputs):
        self.nfe += 1
        return self.ode_func_net_diff(inputs, self._supports)

    def ode_func_net_diff(self, inputs, _supports):
        c = inputs
        for i in range(self._gen_layers):
            c = self._gconv_dif(c, self._num_units, _supports)
            c = self._activation(c)
        c = self._gconv_dif(c, self._latent_dim, _supports)
        c = self._activation(c)
        return c

    def _gconv_dif(self, inputs, output_size, _supports, bias_start=0.0):
        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        input_size = inputs.size(2)
        x = inputs
        x0 = x.permute(1, 2, 0)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, 0)
        if self._gcn_step != 0:
            for support in self._supports:
                x1 = torch.sparse.mm(support, x0)
                x = self._concat(x, x1)
                for k in range(2, self._gcn_step + 1):
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1
        num_matrices = len(self._supports) * self._gcn_step + 1
        x = torch.reshape(x, shape=[num_matrices, self._num_nodes, input_size, batch_size])
        x = x.permute(3, 1, 2, 0)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * num_matrices])
        weights = self._gconv_params.get_weights((input_size * num_matrices, output_size))
        x = torch.matmul(x, weights)
        biases = self._gconv_params.get_biases(output_size, bias_start)
        x += biases
        return torch.reshape(x, [batch_size, self._num_nodes * output_size])

class DiffeqSolver(nn.Module):
    def __init__(self, odefunc, method, latent_dim, odeint_rtol=1e-4, odeint_atol=1e-5):
        super().__init__()
        self.ode_method = method
        self.odefunc = odefunc
        self.latent_dim = latent_dim
        self.rtol = odeint_rtol
        self.atol = odeint_atol

    def forward(self, first_point, time_steps_to_pred):
        n_traj_samples, batch_size = first_point.size()[0], first_point.size()[1]
        first_point = first_point.reshape(n_traj_samples * batch_size, -1)
        self.odefunc.nfe = 0
        pred_y = odeint(self.odefunc, first_point, time_steps_to_pred,
                        rtol=self.rtol, atol=self.atol, method=self.ode_method)
        pred_y = pred_y.reshape(pred_y.size()[0], n_traj_samples, batch_size, -1)
        return pred_y, (self.odefunc.nfe, 0)


class Encoder_z0_RNN(nn.Module):
    def __init__(self, adj_mx, input_dim, rnn_units, latent_dim, node_num, device):
        super().__init__()
        self.rnn_units = rnn_units
        self.latent_dim = latent_dim
        self.num_nodes = node_num
        self.input_dim = input_dim
        self.gru_rnn = GRU(self.input_dim, self.rnn_units).to(device)
        self.hiddens_to_z0 = nn.Sequential(
            nn.Linear(self.rnn_units, 50),
            nn.Tanh(),
            nn.Linear(50, self.latent_dim * 2))
        init_network_weights(self.hiddens_to_z0)

    def forward(self, inputs, batch_size):
        seq_len = inputs.shape[0]
        inputs = inputs.reshape(seq_len, batch_size * self.num_nodes, self.input_dim)
        outputs, _ = self.gru_rnn(inputs)
        last_output = outputs[-1]
        last_output = torch.reshape(last_output, (batch_size, self.num_nodes, self.rnn_units))
        mean, std = split_last_dim(self.hiddens_to_z0(last_output))
        mean = torch.reshape(mean, (1, batch_size, self.num_nodes * self.latent_dim))
        std = torch.reshape(std, (1, batch_size, self.num_nodes * self.latent_dim))
        std = std.abs()
        # Also return wind vars placeholder
        return mean, std, None


class Decoder(nn.Module):
    def __init__(self, output_dim, adj_mx, num_nodes, num_edges):
        super().__init__()
        self.output_dim = output_dim
        self.num_nodes = num_nodes
        self.fc = nn.Linear(4, output_dim)  # latent_dim=4 -> output_dim

    def forward(self, sol_ys):
        # sol_ys: (horizon, n_traj_samples, batch_size, num_nodes * latent_dim)
        horizon, n_traj_samples, batch_size, _ = sol_ys.shape
        sol_ys = sol_ys.reshape(horizon, n_traj_samples, batch_size, self.num_nodes, -1)
        outputs = self.fc(sol_ys)
        # Average over trajectory samples
        outputs = outputs.mean(dim=1)  # (horizon, batch_size, num_nodes, output_dim)
        outputs = outputs.reshape(horizon, batch_size, self.num_nodes * self.output_dim)
        return outputs

class AirPhyNet(nn.Module):
    """AirPhyNet: Physics-informed Neural ODE for air quality forecasting."""

    def __init__(self, adj_mx, node_num, input_dim, output_dim, seq_len, horizon,
                 latent_dim=4, rnn_units=64, ode_method='dopri5', device=None):
        super().__init__()
        if device is None:
            device = torch.device('cpu')
        self.device = device
        self.node_num = node_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.horizon = horizon
        self.latent_dim = latent_dim
        self.n_traj_samples = 3
        self.ode_method = ode_method
        self.atol = 1e-5
        self.rtol = 1e-5
        self.gcn_step = 2

        self.encoder_z0 = Encoder_z0_RNN(adj_mx, input_dim, rnn_units=rnn_units,
                                          latent_dim=latent_dim, node_num=node_num,
                                          device=device)
        self.decoder = Decoder(output_dim, adj_mx, node_num,
                               int((adj_mx > 0.).sum()))
        self.adj_mx = adj_mx

    def forward(self, inputs, labels=None):
        """
        Parameters
        ----------
        inputs : (B, T, N, F)
        Returns
        -------
        (B, horizon, N, output_dim)
        """
        bs = inputs.shape[0]
        # b t n f -> t b n*f
        inputs_reshaped = inputs.transpose(0, 1).reshape(self.seq_len, -1, self.input_dim)

        first_point_mu, first_point_std, _ = self.encoder_z0(inputs_reshaped, bs)

        means_z0 = first_point_mu.repeat(self.n_traj_samples, 1, 1)
        sigma_z0 = first_point_std.repeat(self.n_traj_samples, 1, 1)
        first_point_enc = sample_standard_gaussian(means_z0, sigma_z0)

        time_steps_to_predict = torch.arange(start=0, end=self.horizon, step=1).float().to(self.device)
        time_steps_to_predict = time_steps_to_predict / len(time_steps_to_predict)

        odefunc = ODEFunc(64, self.latent_dim, self.adj_mx, self.gcn_step,
                          self.node_num, filter_type='diff', device=self.device).to(self.device)
        diffeq_solver = DiffeqSolver(odefunc, self.ode_method, self.latent_dim,
                                      odeint_rtol=self.rtol, odeint_atol=self.atol)
        sol_ys, fe = diffeq_solver(first_point_enc, time_steps_to_predict)

        outputs = self.decoder(sol_ys)
        # outputs: (horizon, batch_size, num_nodes * output_dim)
        outputs = outputs.transpose(0, 1).reshape(-1, self.horizon, self.node_num, self.output_dim)
        return outputs
