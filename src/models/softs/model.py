"""SOFTS model implementation.

Vendored/adapted from https://github.com/thuml/Time-Series-Library
(models/SOFTS.py), MIT License.

SOFTS: Series-cOre Fused Time Series forecaster (NeurIPS 2024). Uses an
inverted (variate-as-token) embedding and a stack of STAR (STar
Aggregate-Redistribute) blocks that fuse a learned global "series core" back
into each channel via an MLP, replacing pairwise self-attention.

Adapted for ModernTSF: the upstream ``configs``-object constructor is replaced
with plain keyword arguments, and the shared layers under ``models.module.*``
are reused (``DataEmbedding_inverted``, ``Encoder``, ``EncoderLayer``). The
``EncoderLayer`` consumes the local ``STAR`` module in place of an attention
layer: its ``forward(input, *args, **kwargs)`` returns ``(output, None)`` so it
slots directly into the shared encoder's ``attention(x, x, x, ...)`` call.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.module.embed import DataEmbedding_inverted
from models.module.transformer_encdec import Encoder, EncoderLayer


class STAR(nn.Module):
    """STar Aggregate-Redistribute module (series-core fusion MLP)."""

    def __init__(self, d_series, d_core):
        super().__init__()
        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)
        self.gen3 = nn.Linear(d_series + d_core, d_series)
        self.gen4 = nn.Linear(d_series, d_series)

    def forward(self, input, *args, **kwargs):
        batch_size, channels, d_series = input.shape

        # set FFN
        combined_mean = F.gelu(self.gen1(input))
        combined_mean = self.gen2(combined_mean)

        # stochastic pooling
        if self.training:
            ratio = F.softmax(combined_mean, dim=1)
            ratio = ratio.permute(0, 2, 1)
            ratio = ratio.reshape(-1, channels)
            indices = torch.multinomial(ratio, 1)
            indices = indices.view(batch_size, -1, 1).permute(0, 2, 1)
            combined_mean = torch.gather(combined_mean, 1, indices)
            combined_mean = combined_mean.repeat(1, channels, 1)
        else:
            weight = F.softmax(combined_mean, dim=1)
            combined_mean = torch.sum(
                combined_mean * weight, dim=1, keepdim=True
            ).repeat(1, channels, 1)

        # mlp fusion
        combined_mean_cat = torch.cat([input, combined_mean], -1)
        combined_mean_cat = F.gelu(self.gen3(combined_mean_cat))
        combined_mean_cat = self.gen4(combined_mean_cat)
        output = combined_mean_cat

        return output, None


class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        features="M",
        label_len=0,
        d_model=128,
        d_core=64,
        d_ff=256,
        e_layers=2,
        dropout=0.1,
        activation="gelu",
        use_norm=True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.use_norm = use_norm

        # Embedding (inverted: variate-as-token)
        self.enc_embedding = DataEmbedding_inverted(seq_len, d_model, dropout=dropout)

        # Encoder: STAR blocks fuse a global series core back into each channel
        self.encoder = Encoder(
            [
                EncoderLayer(
                    STAR(d_model, d_core),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
        )

        # Decoder
        self.projection = nn.Linear(d_model, pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization from Non-stationary Transformer
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc /= stdev

        _, _, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        # De-Normalization from Non-stationary Transformer
        if self.use_norm:
            dec_out = dec_out * (
                stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )
            dec_out = dec_out + (
                means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )
        return dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len :, :]  # [B, L, D]
