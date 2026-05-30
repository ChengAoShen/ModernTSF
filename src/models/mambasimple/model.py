"""MambaSimple model implementation.

Vendored/adapted from https://github.com/thuml/Time-Series-Library
(models/MambaSimple.py), MIT License.

Mamba: Linear-Time Sequence Modeling with Selective State Spaces
(https://arxiv.org/abs/2312.00752). This is the dependency-FREE variant: the
selective scan is implemented manually in pure PyTorch (sequential recurrence),
so it does NOT require the ``mamba_ssm`` / ``causal-conv1d`` CUDA kernels.
Implementation reference: https://github.com/johnma2006/mamba-minimal/

Adapted for ModernTSF: the upstream ``configs``-object constructor is replaced
with plain keyword arguments, the non-forecasting task branches are dropped, and
the shared ``DataEmbedding`` layer under ``models.module.embed`` is reused. The
``ResidualBlock`` / ``MambaBlock`` / ``RMSNorm`` blocks are Mamba-specific and
are kept local to this file.

Note (faithful to upstream): the selective-scan state dimension ``n`` is driven
by the ``d_ff`` argument, matching the upstream MambaSimple implementation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat

from models.module.embed import DataEmbedding


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = (
            x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        )
        return output


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_inner, dt_rank, d_conv, d_ff):
        super().__init__()
        self.d_inner = d_inner
        self.dt_rank = dt_rank
        self.d_ff = d_ff

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
        )

        # takes in x and outputs the input-specific delta, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_ff * 2, bias=False)

        # projects delta
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(torch.arange(1, d_ff + 1), "n -> d n", d=self.d_inner).float()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """Figure 3 in Section 3.4 in the paper."""
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)  # [B, L, 2 * d_inner]
        (x, res) = x_and_res.split(
            split_size=[self.d_inner, self.d_inner], dim=-1
        )

        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, "b d l -> b l d")

        x = F.silu(x)

        y = self.ssm(x)
        y = y * F.silu(res)

        output = self.out_proj(y)
        return output

    def ssm(self, x):
        """Algorithm 2 in Section 3.2 in the paper."""
        (d_in, n) = self.A_log.shape

        A = -torch.exp(self.A_log.float())  # [d_in, n]
        D = self.D.float()  # [d_in]

        x_dbl = self.x_proj(x)  # [B, L, d_rank + 2 * d_ff]
        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))  # [B, L, d_in]
        y = self.selective_scan(x, delta, A, B, C, D)

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]

        # A is discretized using zero-order hold (ZOH) discretization
        deltaA = torch.exp(einsum(delta, A, "b l d, d n -> b l d n"))
        # B is discretized using a simplified Euler discretization
        deltaB_u = einsum(delta, B, u, "b l d, b l n, b l d -> b l d n")

        # selective scan, sequential instead of parallel
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], "b d n, b n -> b d")
            ys.append(y)

        y = torch.stack(ys, dim=1)  # [B, L, d_in]
        y = y + u * D

        return y


class ResidualBlock(nn.Module):
    def __init__(self, d_model, d_inner, dt_rank, d_conv, d_ff):
        super().__init__()
        self.mixer = MambaBlock(d_model, d_inner, dt_rank, d_conv, d_ff)
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        output = self.mixer(self.norm(x)) + x
        return output


class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        c_out=None,
        features="M",
        d_model=128,
        d_ff=16,
        e_layers=2,
        expand=2,
        d_conv=4,
        dropout=0.1,
        embed="timeF",
        freq="h",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features
        c_out = c_out if c_out is not None else enc_in

        self.d_inner = d_model * expand
        self.dt_rank = math.ceil(d_model / 16)

        self.embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)

        self.layers = nn.ModuleList(
            [
                ResidualBlock(d_model, self.d_inner, self.dt_rank, d_conv, d_ff)
                for _ in range(e_layers)
            ]
        )
        self.norm = RMSNorm(d_model)

        self.out_layer = nn.Linear(d_model, c_out, bias=False)

    def forecast(self, x_enc, x_mark_enc):
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x_enc = x_enc / std_enc

        x = self.embedding(x_enc, x_mark_enc)
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)
        x_out = self.out_layer(x)

        x_out = x_out * std_enc + mean_enc
        return x_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        x_out = self.forecast(x_enc, x_mark_enc)
        return x_out[:, -self.pred_len :, :]  # [B, pred_len, c_out]
