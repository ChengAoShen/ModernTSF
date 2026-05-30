"""BiMamba model implementation.

Architecture adapted from https://github.com/Huangmr0719/BiMamba
(BiMamba.py) -- no license file declared in the upstream repo (unlicensed).
Bidirectional Mamba following "Bi-Mamba+: Bidirectional Mamba for Time Series
Forecasting" (https://arxiv.org/abs/2404.15772): each block runs a Mamba scan
over the sequence forward and over the time-reversed sequence, then averages
the two branches (each with its own Add&Norm + FeedForward).

IMPORTANT: the upstream ``BiMambaBlock`` depends on ``mamba_ssm.Mamba`` (CUDA
kernels). To stay dependency-free, this port reuses the kernel-free selective
scan ``MambaBlock`` vendored in ``models.mambasimple.model`` (itself MIT, from
thuml/Time-Series-Library MambaSimple.py + mamba-minimal). The bidirectional
wrapper (forward + flipped scan, Add&Norm, FFN, branch averaging) is
re-implemented locally around that kernel-free block.

Adapted for ModernTSF: the upstream stand-alone ``BiMambaEncoder`` is wrapped in
a TSLib-style ``Model`` with a plain-kwargs constructor, the shared
``DataEmbedding`` layer under ``models.module.embed`` is reused, Non-stationary
instance normalisation is applied, and a flatten/linear head maps the encoded
sequence to the forecast horizon. Only the long-term forecast path is kept.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.mambasimple.model import MambaBlock
from models.module.embed import DataEmbedding


class FeedForward(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = d_model * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class BiMambaBlock(nn.Module):
    """Bidirectional Mamba block: forward + time-reversed scan, then averaged.

    Uses the kernel-free ``MambaBlock`` instead of ``mamba_ssm.Mamba``.
    """

    def __init__(
        self,
        d_model: int,
        d_inner: int,
        dt_rank: int,
        d_conv: int = 4,
        d_state: int = 16,
        dropout: float = 0.1,
        share_ffn: bool = False,
        share_norm: bool = False,
    ):
        super().__init__()

        self.mamba_fwd = MambaBlock(d_model, d_inner, dt_rank, d_conv, d_state)
        self.mamba_rev = MambaBlock(d_model, d_inner, dt_rank, d_conv, d_state)

        if share_norm:
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.ln1_rev = self.ln1
            self.ln2_rev = self.ln2
        else:
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            self.ln1_rev = nn.LayerNorm(d_model)
            self.ln2_rev = nn.LayerNorm(d_model)

        if share_ffn:
            self.ffn = FeedForward(d_model, dropout=dropout)
            self.ffn_rev = self.ffn
        else:
            self.ffn = FeedForward(d_model, dropout=dropout)
            self.ffn_rev = FeedForward(d_model, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward_branch(self, x, mamba, ln1, ln2, ffn, flip_time=False):
        x_in = torch.flip(x, dims=[1]) if flip_time else x

        y = mamba(x_in)
        y = self.dropout(y)

        if flip_time:
            y = torch.flip(y, dims=[1])
        y = ln1(x + y)  # Add & Norm

        y2 = ffn(y)  # FeedForward
        y2 = self.dropout(y2)
        y = ln2(y + y2)  # Add & Norm
        return y

    def forward(self, x):
        """x: (B, S, D) -> (B, S, D)."""
        out_fwd = self.forward_branch(
            x, self.mamba_fwd, self.ln1, self.ln2, self.ffn, flip_time=False
        )
        out_rev = self.forward_branch(
            x, self.mamba_rev, self.ln1_rev, self.ln2_rev, self.ffn_rev, flip_time=True
        )
        return 0.5 * (out_fwd + out_rev)


class BiMambaEncoder(nn.Module):
    def __init__(self, d_model: int, num_layers: int, **block_kwargs):
        super().__init__()
        self.layers = nn.ModuleList(
            [BiMambaBlock(d_model=d_model, **block_kwargs) for _ in range(num_layers)]
        )
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.final_ln(x)


class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        c_out=None,
        features="M",
        d_model=128,
        d_state=16,
        e_layers=2,
        expand=2,
        d_conv=4,
        dropout=0.1,
        share_ffn=False,
        share_norm=False,
        embed="timeF",
        freq="h",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features
        c_out = c_out if c_out is not None else enc_in
        self.c_out = c_out

        self.d_inner = d_model * expand
        self.dt_rank = math.ceil(d_model / 16)

        self.embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)

        self.encoder = BiMambaEncoder(
            d_model=d_model,
            num_layers=e_layers,
            d_inner=self.d_inner,
            dt_rank=self.dt_rank,
            d_conv=d_conv,
            d_state=d_state,
            dropout=dropout,
            share_ffn=share_ffn,
            share_norm=share_norm,
        )

        # Map encoded sequence (seq_len, d_model) -> forecast (pred_len, c_out).
        self.projection = nn.Linear(d_model, c_out, bias=True)
        self.temporal = nn.Linear(seq_len, pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc):
        # Non-stationary instance normalisation.
        mean_enc = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x_enc = x_enc / std_enc

        x = self.embedding(x_enc, x_mark_enc)  # [B, seq_len, d_model]
        x = self.encoder(x)  # [B, seq_len, d_model]

        x = self.projection(x)  # [B, seq_len, c_out]
        x = self.temporal(x.transpose(1, 2)).transpose(1, 2)  # [B, pred_len, c_out]

        # De-normalise using the (broadcast) last-channel stats, matching the
        # MambaSimple convention; std/mean are [B, 1, enc_in].
        std_last = std_enc[:, :, : self.c_out]
        mean_last = mean_enc[:, :, : self.c_out]
        x = x * std_last + mean_last
        return x

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        x_out = self.forecast(x_enc, x_mark_enc)
        return x_out[:, -self.pred_len :, :]  # [B, pred_len, c_out]
