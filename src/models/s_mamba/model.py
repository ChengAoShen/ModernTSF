"""S-Mamba model implementation.

Vendored/adapted from https://github.com/wzhwzhwzh0921/S-D-Mamba
(model/S_Mamba.py and layers/Mamba_EncDec.py), the official code for
"Is Mamba Effective for Time Series Forecasting?" (https://arxiv.org/abs/2403.11144).
The upstream repository ships no explicit LICENSE file; the architecture is an
iTransformer-style inverted embedding followed by a bidirectional Mamba
encoder. The licenses of related iTransformer / Time-Series-Library code do
not grant a license for this author repository, so its codebase is retained as
reference-only metadata.

S-Mamba delegates inter-variate correlation extraction to a bidirectional Mamba
block (over the variate/token axis) and temporal dependencies to a Feed-Forward
network, on top of the inverted (variate-as-token) embedding.

Adapted for ModernTSF:
- The upstream ``configs``-object constructor is replaced with plain keyword
  arguments and the non-forecasting branches are dropped (long-term forecast only).
- Upstream imports ``mamba_ssm`` CUDA kernels. This implementation uses the
  repository-wide pure-PyTorch ``MambaBlock`` so the same selective scan runs
  on CPU and GPU without a second model-local implementation.
- The shared ``DataEmbedding_inverted`` layer under ``components.embed`` is
  reused. The ``Encoder`` / ``EncoderLayer`` are S-Mamba specific and kept local.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from components.embed import DataEmbedding_inverted
from components.mamba import MambaBlock


def _mamba_block(
    d_model: int, d_state: int, d_conv: int, expand: int
) -> MambaBlock:
    """Map the paper-facing Mamba parameters to the shared block contract."""
    return MambaBlock(
        d_model=d_model,
        d_inner=int(expand * d_model),
        dt_rank=math.ceil(d_model / 16),
        d_conv=d_conv,
        d_state=d_state,
    )


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
                    _mamba_block(
                        d_model=d_model,
                        d_state=d_state,
                        d_conv=d_conv,
                        expand=expand,
                    ),
                    _mamba_block(
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
