"""S-Mamba model implementation.

Vendored/adapted from https://github.com/wzhwzhwzh0921/S-D-Mamba
(model/S_Mamba.py and layers/Mamba_EncDec.py), the official code for
"Is Mamba Effective for Time Series Forecasting?" (https://arxiv.org/abs/2403.11144).
The upstream repository ships no explicit LICENSE file; the architecture is an
iTransformer-style inverted embedding (iTransformer is MIT, thuml/Time-Series-Library)
followed by a bidirectional Mamba encoder, and is treated as MIT here per the
author/TSLib provenance.

S-Mamba delegates inter-variate correlation extraction to a bidirectional Mamba
block (over the variate/token axis) and temporal dependencies to a Feed-Forward
network, on top of the inverted (variate-as-token) embedding.

Adapted for ModernTSF:
- The upstream ``configs``-object constructor is replaced with plain keyword
  arguments and the non-forecasting branches are dropped (long-term forecast only).
- CRITICAL: upstream imports ``mamba_ssm`` (CUDA selective-scan kernels), which is
  not installable on CPU/macOS. The ``Mamba`` block below is a dependency-FREE,
  pure-PyTorch selective scan reused from the already-ported
  ``src/models/mambasimple/model.py`` (kernel-free ``MambaBlock``/``RMSNorm``,
  reference https://github.com/johnma2006/mamba-minimal). It mirrors the
  ``mamba_ssm.Mamba(d_model, d_state, d_conv, expand)`` constructor signature so
  the upstream encoder wiring is preserved.
- The shared ``DataEmbedding_inverted`` layer under ``models.module.embed`` is
  reused. The ``Encoder`` / ``EncoderLayer`` are S-Mamba specific and kept local.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import einsum, rearrange, repeat

from models.module.embed import DataEmbedding_inverted


class Mamba(nn.Module):
    """Dependency-free Mamba block (kernel-free selective scan).

    Mirrors the ``mamba_ssm.Mamba(d_model, d_state, d_conv, expand)`` API used by
    upstream S-Mamba, but performs the selective scan sequentially in pure PyTorch
    (no ``mamba_ssm`` / ``causal-conv1d`` CUDA kernels). Logic ported from
    ``src/models/mambasimple/model.py::MambaBlock`` with ``d_ff`` -> ``d_state``.
    """

    def __init__(self, d_model, d_state=16, d_conv=2, expand=1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)

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
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False
        )
        # projects delta
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(
            torch.arange(1, self.d_state + 1), "n -> d n", d=self.d_inner
        ).float()
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
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

        return self.out_proj(y)

    def ssm(self, x):
        (d_in, n) = self.A_log.shape

        A = -torch.exp(self.A_log.float())  # [d_in, n]
        D = self.D.float()  # [d_in]

        x_dbl = self.x_proj(x)  # [B, L, dt_rank + 2 * d_state]
        (delta, B, C) = x_dbl.split(
            split_size=[self.dt_rank, n, n], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))  # [B, L, d_in]
        return self.selective_scan(x, delta, A, B, C, D)

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]

        deltaA = torch.exp(einsum(delta, A, "b l d, d n -> b l d n"))
        deltaB_u = einsum(delta, B, u, "b l d, b l n, b l d -> b l d n")

        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], "b d n, b n -> b d")
            ys.append(y)

        y = torch.stack(ys, dim=1)  # [B, L, d_in]
        y = y + u * D
        return y


class EncoderLayer(nn.Module):
    def __init__(
        self, attention, attention_r, d_model, d_ff=None, dropout=0.1, activation="relu"
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.attention_r = attention_r
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # bidirectional Mamba over the variate-token axis
        new_x = self.attention(x) + self.attention_r(x.flip(dims=[1])).flip(dims=[1])
        attn = None

        x = x + new_x
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class Model(nn.Module):
    """S-Mamba: inverted embedding + bidirectional Mamba encoder."""

    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        features="M",
        d_model=128,
        d_state=16,
        d_ff=128,
        e_layers=2,
        d_conv=2,
        expand=1,
        dropout=0.1,
        activation="gelu",
        use_norm=True,
        embed="timeF",
        freq="h",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features
        self.use_norm = use_norm

        self.enc_embedding = DataEmbedding_inverted(
            seq_len, d_model, embed, freq, dropout
        )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    Mamba(
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                    ),
                    Mamba(
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=nn.LayerNorm(d_model),
        )
        self.projector = nn.Linear(d_model, pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc = x_enc / stdev

        _, _, N = x_enc.shape  # B L N

        # B L N -> B N E
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        # B N E -> B N S -> B S N
        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :N]

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (
                stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )
            dec_out = dec_out + (
                means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )

        return dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len :, :]  # [B, pred_len, c_out]
