"""iTransformer model implementation."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.module.embed import DataEmbedding_inverted
from models.module.self_attention_family import AttentionLayer, FullAttention
from models.module.transformer_encdec import Encoder, EncoderLayer


class ITransformerModel(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        output_attention,
        use_norm,
        d_model,
        embed,
        freq,
        class_strategy,
        factor,
        n_heads,
        d_ff,
        dropout,
        activation,
        e_layers,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.use_norm = use_norm
        self.enc_embedding = DataEmbedding_inverted(
            seq_len,
            d_model,
            embed,
            freq,
            dropout,
        )
        self.class_strategy = class_strategy
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=output_attention,
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
        self.projector = nn.Linear(d_model, pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc /= stdev

        _, _, num_vars = x_enc.shape

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projector(enc_out).permute(0, 2, 1)[:, :, :num_vars]

        if self.use_norm:
            dec_out = dec_out * (
                stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )
            dec_out = dec_out + (
                means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

        if self.output_attention:
            return dec_out[:, -self.pred_len :, :], attns
        return dec_out[:, -self.pred_len :, :]


class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        d_model,
        n_heads,
        e_layers,
        d_ff,
        factor,
        dropout,
        embed,
        activation,
        output_attention,
        use_norm,
        freq,
        class_strategy,
    ):
        super().__init__()
        self.model = ITransformerModel(
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            factor=factor,
            dropout=dropout,
            embed=embed,
            activation=activation,
            output_attention=output_attention,
            use_norm=use_norm,
            freq=freq,
            class_strategy=class_strategy,
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.model(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
