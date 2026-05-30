"""Informer model implementation.

Vendored/adapted from https://github.com/thuml/Time-Series-Library
(models/Informer.py), MIT License.

Informer: Beyond Efficient Transformer for Long Sequence Time-Series
Forecasting (AAAI 2021).

Adapted for ModernTSF: the upstream ``configs``-object constructor is replaced
with plain keyword arguments, and the shared layers under ``models.module.*``
are reused (``DataEmbedding``, ``ProbAttention``, ``AttentionLayer``, and the
composite ``Encoder`` / ``EncoderLayer`` / ``Decoder`` / ``DecoderLayer`` /
``ConvLayer`` from ``transformer_encdec``). Only the long-term forecast path is
kept; the classification / imputation / anomaly-detection branches are dropped.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.module.embed import DataEmbedding
from models.module.self_attention_family import AttentionLayer, ProbAttention
from models.module.transformer_encdec import (
    ConvLayer,
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
)


class Model(nn.Module):
    """Informer with ProbSparse attention in O(L log L) complexity."""

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
        distil=True,
        embed="timeF",
        freq="h",
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.features = features

        if dec_in is None:
            dec_in = enc_in
        if c_out is None:
            c_out = 1 if features == "MS" else enc_in

        # Embedding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(
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
            [ConvLayer(d_model) for _ in range(e_layers - 1)] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        ProbAttention(
                            True,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    AttentionLayer(
                        ProbAttention(
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

    def long_forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        return dec_out  # [B, L, D]

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.long_forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len :, :]  # [B, L, D]
