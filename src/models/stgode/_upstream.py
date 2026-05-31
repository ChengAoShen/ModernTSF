"""Verbatim STGODE model source.

Vendored from CauAir (src/models/stgode.py).
BaseModel replaced with nn.Module; explicit params stored on self.
Reference: https://github.com/square-coder/STGODE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            conv = nn.Conv2d(in_channels, out_channels, (1, kernel_size),
                             dilation=(1, dilation_size), padding=(0, padding))
            conv.weight.data.normal_(0, 0.01)
            chomp = Chomp1d(padding)
            relu = nn.ReLU()
            drop = nn.Dropout(dropout)
            layers += [nn.Sequential(conv, chomp, relu, drop)]

        self.network = nn.Sequential(*layers)
        self.downsample = nn.Conv2d(num_inputs, num_channels[-1], (1, 1)) \
            if num_inputs != num_channels[-1] else None
        if self.downsample:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y = x.permute(0, 3, 1, 2)
        y = F.relu(self.network(y) + self.downsample(y) if self.downsample else y)
        y = y.permute(0, 2, 3, 1)
        return y


class ODEFunc(nn.Module):
    def __init__(self, feature_dim, temporal_dim, adj):
        super(ODEFunc, self).__init__()
        self.adj = adj
        self.x0 = None
        self.alpha = nn.Parameter(0.8 * torch.ones(adj.shape[1]))
        self.beta = 0.6
        self.w = nn.Parameter(torch.eye(feature_dim))
        self.d = nn.Parameter(torch.zeros(feature_dim) + 1)
        self.w2 = nn.Parameter(torch.eye(temporal_dim))
        self.d2 = nn.Parameter(torch.zeros(temporal_dim) + 1)

    def forward(self, t, x):
        alpha = torch.sigmoid(self.alpha).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        xa = torch.einsum('ij, kjlm->kilm', self.adj, x)
        d = torch.clamp(self.d, min=0, max=1)
        w = torch.mm(self.w * d, torch.t(self.w))
        xw = torch.einsum('ijkl, lm->ijkm', x, w)
        d2 = torch.clamp(self.d2, min=0, max=1)
        w2 = torch.mm(self.w2 * d2, torch.t(self.w2))
        xw2 = torch.einsum('ijkl, km->ijml', x, w2)
        f = alpha / 2 * xa - x + xw - x + xw2 - x + self.x0
        return f


class ODEblock(nn.Module):
    def __init__(self, odefunc, t=torch.tensor([0, 1])):
        super(ODEblock, self).__init__()
        self.t = t
        self.odefunc = odefunc

    def set_x0(self, x0):
        self.odefunc.x0 = x0.clone().detach()

    def forward(self, x):
        t = self.t.type_as(x)
        z = odeint(self.odefunc, x, t, method='euler')[1]
        return z


class ODEG(nn.Module):
    def __init__(self, feature_dim, temporal_dim, adj, time):
        super(ODEG, self).__init__()
        self.odeblock = ODEblock(ODEFunc(feature_dim, temporal_dim, adj),
                                 t=torch.tensor([0, time]))

    def forward(self, x):
        self.odeblock.set_x0(x)
        z = self.odeblock(x)
        return F.relu(z)


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, node_num, A_hat, seq_len):
        super(STGCNBlock, self).__init__()
        self.A_hat = A_hat
        self.temporal1 = TemporalConvNet(num_inputs=in_channels,
                                         num_channels=out_channels)
        self.odeg = ODEG(out_channels[-1], seq_len, A_hat, time=6)
        self.temporal2 = TemporalConvNet(num_inputs=out_channels[-1],
                                         num_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(node_num)

    def forward(self, X):
        t = self.temporal1(X)
        t = self.odeg(t)
        t = self.temporal2(F.relu(t))
        return self.batch_norm(t)


class STGODE(nn.Module):
    """Spatial-Temporal Graph ODE Network."""

    def __init__(self, A_sp, A_se, node_num, input_dim, output_dim,
                 seq_len, horizon, num_layers=3):
        super(STGODE, self).__init__()
        self.node_num = node_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.horizon = horizon

        self.sp_blocks = nn.ModuleList([
            nn.Sequential(
                STGCNBlock(in_channels=input_dim, out_channels=[64, 32, 64],
                           node_num=node_num, A_hat=A_sp, seq_len=seq_len),
                STGCNBlock(in_channels=64, out_channels=[64, 32, 64],
                           node_num=node_num, A_hat=A_sp, seq_len=seq_len)
            ) for _ in range(num_layers)
        ])

        self.se_blocks = nn.ModuleList([
            nn.Sequential(
                STGCNBlock(in_channels=input_dim, out_channels=[64, 32, 64],
                           node_num=node_num, A_hat=A_se, seq_len=seq_len),
                STGCNBlock(in_channels=64, out_channels=[64, 32, 64],
                           node_num=node_num, A_hat=A_se, seq_len=seq_len)
            ) for _ in range(num_layers)
        ])

        self.pred = nn.Sequential(
            nn.Linear(seq_len * 64, horizon * 32),
            nn.ReLU(),
            nn.Linear(horizon * 32, horizon)
        )

    def forward(self, x, label=None):
        """Forward pass. x: (B, T, N, F) -> (B, horizon, N, 1)"""
        x = x.transpose(1, 2)  # (B, N, T, F)
        outs = []
        for blk in self.sp_blocks:
            outs.append(blk(x))
        for blk in self.se_blocks:
            outs.append(blk(x))
        outs = torch.stack(outs)
        x = torch.max(outs, dim=0)[0]
        x = x.reshape((x.shape[0], x.shape[1], -1))
        x = self.pred(x)
        x = x.unsqueeze(-1).transpose(1, 2)  # (B, horizon, N, 1)
        return x
