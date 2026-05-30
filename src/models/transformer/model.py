"""Vanilla Transformer model implementation.

Vendored/adapted from https://github.com/thuml/Time-Series-Library
(models/Transformer.py), MIT License.

Vanilla encoder-decoder Transformer with O(L^2) self-attention
(Vaswani et al., 2017), applied to long-term time-series forecasting.

Adapted for ModernTSF: the upstream ``configs``-object constructor is replaced
with plain keyword arguments, and the shared layers under ``models.module.*``
are reused (``DataEmbedding``, ``FullAttention``, ``AttentionLayer`` and the
composite ``Encoder`` / ``EncoderLayer`` / ``Decoder`` / ``DecoderLayer``
blocks). Non-forecasting task branches (imputation / anomaly / classification)
are dropped; only the long-term forecast path is kept.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.module.embed import DataEmbedding
from models.module.self_attention_family import AttentionLayer, FullAttention
from models.module.transformer_encdec import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
)


class Model(nn.Module):
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
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.features = features

        dec_in = dec_in if dec_in is not None else enc_in
        c_out = c_out if c_out is not None else (1 if features == "MS" else enc_in)

        # Embedding
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
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
                        FullAttention(
                            True,
                            factor,
                            attention_dropout=dropout,
                            output_attention=False,
                        ),
                        d_model,
                        n_heads,
                    ),
                    AttentionLayer(
                        FullAttention(
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

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)
        return dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len :, :]  # [B, L, D]
