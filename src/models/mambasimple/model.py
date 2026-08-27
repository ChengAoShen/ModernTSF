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
the shared embedding and kernel-free state-space blocks under ``components``
are reused by all compatible named models.

Note (faithful to upstream): the selective-scan state dimension ``n`` is driven
by the ``d_ff`` argument, matching the upstream MambaSimple implementation.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from components.embed import DataEmbedding
from components.mamba import MambaResidualBlock, RMSNorm


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
                MambaResidualBlock(d_model, self.d_inner, self.dt_rank, d_conv, d_ff)
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
