"""Vendored STGODE (Spatial-Temporal Graph ODE Network) architecture.

Vendored/adapted from https://github.com/GestaltCogTeam/BasicTS
(baselines/STGODE), Apache-2.0.

Two upstream files are merged here: ``arch/odegcn.py`` (the neural-ODE graph
convolution block) and ``arch/stgode_arch.py`` (the dual-graph backbone). Two
adaptations were made relative to upstream:

* The ``torchdiffeq`` dependency is removed. Upstream solved the ODE block with
  ``torchdiffeq.odeint(..., method='euler')`` over the interval ``[0, time]``.
  We vendor an equivalent fixed-step explicit Euler integrator
  (:func:`euler_odeint`) that takes a single step of size ``time`` -- matching
  the upstream call ``odeint(func, x0, t=[0, time], method='euler')[1]`` which
  also uses a single Euler step from ``0`` to ``time``.
* Every hardcoded ``.to('cuda')`` / device assumption is removed. Internally
  created tensors and the adjacency matrices follow the input tensor's device
  (see :meth:`ODEFunc.forward`).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def euler_odeint(func, x0: torch.Tensor, time: float) -> torch.Tensor:
    """Single fixed-step explicit Euler integration of ``dx/dt = func(t, x)``.

    Mirrors the upstream ``torchdiffeq.odeint(func, x0, t=[0, time],
    method='euler')[1]`` call, which integrates from ``0`` to ``time`` in one
    Euler step and returns the value at ``time``.

    Parameters
    ----------
    func : callable
        The ODE function ``func(t, x) -> dx/dt``.
    x0 : torch.Tensor
        Initial state at ``t = 0``.
    time : float
        Integration end time (the single step size).

    Returns
    -------
    torch.Tensor
        State at ``t = time``.
    """
    t0 = x0.new_zeros(())
    return x0 + time * func(t0, x0)


# --- odegcn.py ---------------------------------------------------------------


class ODEFunc(nn.Module):
    """Neural-ODE function defining the graph-convolution dynamics."""

    def __init__(self, feature_dim: int, temporal_dim: int, adj: torch.Tensor):
        super().__init__()
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
        xa = torch.einsum("ij, kjlm->kilm", self.adj.to(x.device), x)

        # ensure the eigenvalues to be less than 1
        d = torch.clamp(self.d, min=0, max=1)
        w = torch.mm(self.w * d, torch.t(self.w))
        xw = torch.einsum("ijkl, lm->ijkm", x, w)

        d2 = torch.clamp(self.d2, min=0, max=1)
        w2 = torch.mm(self.w2 * d2, torch.t(self.w2))
        xw2 = torch.einsum("ijkl, km->ijml", x, w2)

        f = alpha / 2 * xa - x + xw - x + xw2 - x + self.x0
        return f


class ODEblock(nn.Module):
    """Wraps an :class:`ODEFunc` and integrates it over ``[0, time]``."""

    def __init__(self, odefunc: ODEFunc, time: float = 1.0):
        super().__init__()
        self.time = float(time)
        self.odefunc = odefunc

    def set_x0(self, x0: torch.Tensor) -> None:
        self.odefunc.x0 = x0.clone().detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return euler_odeint(self.odefunc, x, self.time)


class ODEG(nn.Module):
    """Graph-ODE block: integrate the dynamics then apply ReLU."""

    def __init__(self, feature_dim: int, temporal_dim: int, adj: torch.Tensor, time: float):
        super().__init__()
        self.odeblock = ODEblock(ODEFunc(feature_dim, temporal_dim, adj), time=time)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.odeblock.set_x0(x)
        z = self.odeblock(x)
        return F.relu(z)


# --- stgode_arch.py ----------------------------------------------------------


class Chomp1d(nn.Module):
    """Remove the extra dimension introduced by causal padding."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, : -self.chomp_size].contiguous()


class TemporalConvNet(nn.Module):
    """Dilated temporal convolution stack (operates over the time axis)."""

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            conv = nn.Conv2d(
                in_channels,
                out_channels,
                (1, kernel_size),
                dilation=(1, dilation_size),
                padding=(0, padding),
            )
            conv.weight.data.normal_(0, 0.01)
            chomp = Chomp1d(padding)
            relu = nn.ReLU()
            dropout_layer = nn.Dropout(dropout)
            layers += [nn.Sequential(conv, chomp, relu, dropout_layer)]

        self.network = nn.Sequential(*layers)
        self.downsample = (
            nn.Conv2d(num_inputs, num_channels[-1], (1, 1))
            if num_inputs != num_channels[-1]
            else None
        )
        if self.downsample:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # x : (B, N, T, F) -> permute to (B, F, N, T)
        y = x.permute(0, 3, 1, 2)
        y = F.relu(self.network(y) + self.downsample(y) if self.downsample else y)
        y = y.permute(0, 2, 3, 1)
        return y


class GCN(nn.Module):
    def __init__(self, A_hat, in_channels, out_channels):
        super().__init__()
        self.A_hat = A_hat
        self.theta = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
        self.reset()

    def reset(self):
        stdv = 1.0 / math.sqrt(self.theta.shape[1])
        self.theta.data.uniform_(-stdv, stdv)

    def forward(self, X):
        y = torch.einsum("ij, kjlm-> kilm", self.A_hat.to(X.device), X)
        return F.relu(torch.einsum("kjlm, mn->kjln", y, self.theta))


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, A_hat, temporal_dim):
        super().__init__()
        self.A_hat = A_hat
        self.temporal1 = TemporalConvNet(
            num_inputs=in_channels, num_channels=out_channels
        )
        self.odeg = ODEG(out_channels[-1], temporal_dim, A_hat, time=6)
        self.temporal2 = TemporalConvNet(
            num_inputs=out_channels[-1], num_channels=out_channels
        )
        self.batch_norm = nn.BatchNorm2d(num_nodes)

    def forward(self, X):
        t = self.temporal1(X)
        t = self.odeg(t)
        t = self.temporal2(F.relu(t))
        return self.batch_norm(t)


class ODEGCN(nn.Module):
    """STGODE backbone (dual spatial / semantic graph ODE network).

    Paper: Spatial-Temporal Graph ODE Networks for Traffic Flow Forecasting
    (SIGKDD 2021), https://arxiv.org/abs/2106.12931.
    """

    def __init__(
        self,
        num_nodes,
        num_features,
        num_timesteps_input,
        num_timesteps_output,
        A_sp_hat,
        A_se_hat,
    ):
        super().__init__()
        # The ODE block's temporal_dim must match the time length after the
        # first TemporalConvNet. With causal "same" padding the temporal length
        # is preserved, so it equals num_timesteps_input.
        temporal_dim = num_timesteps_input
        # spatial graph
        self.sp_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    STGCNBlock(
                        in_channels=num_features,
                        out_channels=[64, 32, 64],
                        num_nodes=num_nodes,
                        A_hat=A_sp_hat,
                        temporal_dim=temporal_dim,
                    ),
                    STGCNBlock(
                        in_channels=64,
                        out_channels=[64, 32, 64],
                        num_nodes=num_nodes,
                        A_hat=A_sp_hat,
                        temporal_dim=temporal_dim,
                    ),
                )
                for _ in range(3)
            ]
        )
        # semantic graph
        self.se_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    STGCNBlock(
                        in_channels=num_features,
                        out_channels=[64, 32, 64],
                        num_nodes=num_nodes,
                        A_hat=A_se_hat,
                        temporal_dim=temporal_dim,
                    ),
                    STGCNBlock(
                        in_channels=64,
                        out_channels=[64, 32, 64],
                        num_nodes=num_nodes,
                        A_hat=A_se_hat,
                        temporal_dim=temporal_dim,
                    ),
                )
                for _ in range(3)
            ]
        )

        self.pred = nn.Sequential(
            nn.Linear(num_timesteps_input * 64, num_timesteps_output * 32),
            nn.ReLU(),
            nn.Linear(num_timesteps_output * 32, num_timesteps_output),
        )

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

        Args:
            history_data: input of shape ``(B, T, N, F)``.

        Returns:
            Prediction of shape ``(B, N, T_out, 1)``.
        """
        x = history_data.transpose(1, 2)  # (B, N, T, F)
        outs = []
        for blk in self.sp_blocks:
            outs.append(blk(x))
        for blk in self.se_blocks:
            outs.append(blk(x))
        outs = torch.stack(outs)
        x = torch.max(outs, dim=0)[0]
        x = x.reshape((x.shape[0], x.shape[1], -1))
        x = self.pred(x).transpose(1, 2).unsqueeze(-1)  # (B, N, T_out, 1)
        return x
