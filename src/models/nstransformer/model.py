"""Nonstationary Transformer (NSTransformer) model implementation.

Vendored/adapted from https://github.com/thuml/Time-Series-Library
(models/Nonstationary_Transformer.py), MIT License.

Non-stationary Transformers: Exploring the Stationarity in Time Series
Forecasting (NeurIPS 2022). The model normalises each series, learns
de-stationary factors ``tau`` / ``delta`` via small projector MLPs, and feeds
them into a de-stationary attention so the attention scores are rescaled by the
removed non-stationary statistics.

Adapted for ModernTSF: the upstream ``configs``-object constructor is replaced
with plain keyword arguments, and the shared layers under ``models.module.*``
are reused (``DataEmbedding``, ``AttentionLayer``, the composite ``Encoder`` /
``EncoderLayer`` / ``Decoder`` / ``DecoderLayer``). ``DSAttention`` (de-stationary
attention) and the ``Projector`` MLP are NSTransformer-specific and kept local
to this file. Only the long-term forecasting path is retained.
"""

from __future__ import annotations

from math import sqrt

import numpy as np
import torch
import torch.nn as nn

from models.module.embed import DataEmbedding
from models.module.masking import TriangularCausalMask
from models.module.self_attention_family import AttentionLayer
from models.module.transformer_encdec import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
)


class DSAttention(nn.Module):
    """De-stationary Attention.

    Vendored from layers/SelfAttention_Family.py (thuml/Time-Series-Library).
    Rescales the pre-softmax scores with the learned de-stationary factors
    ``tau`` (multiplicative) and ``delta`` (additive).
    """

    def __init__(
        self,
        mask_flag=True,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
    ):
        super().__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = 0.0 if delta is None else delta.unsqueeze(1).unsqueeze(1)  # B x 1 x 1 x S

        # De-stationary Attention: rescale pre-softmax score with the learned factors
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        return V.contiguous(), None


class Projector(nn.Module):
    """MLP to learn the de-stationary factors.

    Vendored from models/Nonstationary_Transformer.py (thuml/Time-Series-Library).
    """

    def __init__(
        self, enc_in, seq_len, hidden_dims, hidden_layers, output_dim, kernel_size=3
    ):
        super().__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.series_conv = nn.Conv1d(
            in_channels=seq_len,
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )

        layers = [nn.Linear(2 * enc_in, hidden_dims[0]), nn.ReLU()]
        for i in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dims[i], hidden_dims[i + 1]), nn.ReLU()]
        layers += [nn.Linear(hidden_dims[-1], output_dim, bias=False)]
        self.backbone = nn.Sequential(*layers)

    def forward(self, x, stats):
        # x:     B x S x E
        # stats: B x 1 x E
        # y:     B x O
        batch_size = x.shape[0]
        x = self.series_conv(x)  # B x 1 x E
        x = torch.cat([x, stats], dim=1)  # B x 2 x E
        x = x.view(batch_size, -1)  # B x 2E
        y = self.backbone(x)  # B x O
        return y


class Model(nn.Module):
    """Non-stationary Transformer (long-term forecasting only)."""

    def __init__(
        self,
        seq_len,
        pred_len,
        label_len,
        enc_in,
        dec_in=None,
        c_out=None,
        features="M",
        d_model=128,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=256,
        dropout=0.1,
        factor=3,
        activation="gelu",
        embed="timeF",
        freq="h",
        p_hidden_dims=None,
        p_hidden_layers=2,
    ):
        super().__init__()
        self.features = features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.enc_in = enc_in
        dec_in = dec_in if dec_in is not None else enc_in
        # For "M"/"MS"-style forecasting the channel count out equals enc_in here
        # (single-file datasets feed all channels); upstream c_out defaults the same.
        c_out = c_out if c_out is not None else enc_in
        self.c_out = c_out
        if p_hidden_dims is None:
            p_hidden_dims = [128, 128]

        # Embedding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        DSAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )

        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        DSAttention(
                            True,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    AttentionLayer(
                        DSAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
            projection=nn.Linear(d_model, c_out, bias=True),
        )

        self.tau_learner = Projector(
            enc_in=enc_in,
            seq_len=seq_len,
            hidden_dims=p_hidden_dims,
            hidden_layers=p_hidden_layers,
            output_dim=1,
        )
        self.delta_learner = Projector(
            enc_in=enc_in,
            seq_len=seq_len,
            hidden_dims=p_hidden_dims,
            hidden_layers=p_hidden_layers,
            output_dim=seq_len,
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_raw = x_enc.clone().detach()

        # Normalization
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()  # B x 1 x E
        x_enc = x_enc / std_enc

        # B x S x E, B x 1 x E -> B x 1, positive scalar
        tau = self.tau_learner(x_raw, std_enc)
        tau = torch.clamp(tau, max=80.0).exp()  # avoid numerical overflow
        # B x S x E, B x 1 x E -> B x S
        delta = self.delta_learner(x_raw, mean_enc)

        # Build the decoder input: label_len history (normalised) + zero placeholders.
        if self.label_len > 0:
            label_part = x_enc[:, -self.label_len :, :]
        else:
            label_part = x_enc[:, :0, :]
        zero_part = torch.zeros_like(x_dec[:, -self.pred_len :, :])
        x_dec_new = torch.cat([label_part, zero_part], dim=1).to(x_enc.device).clone()

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None, tau=tau, delta=delta)

        dec_out = self.dec_embedding(x_dec_new, x_mark_dec)
        dec_out = self.decoder(
            dec_out, enc_out, x_mask=None, cross_mask=None, tau=tau, delta=delta
        )
        dec_out = dec_out * std_enc + mean_enc
        return dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len :, :]  # [B, L, D]
