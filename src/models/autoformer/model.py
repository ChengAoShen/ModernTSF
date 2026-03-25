"""Autoformer model implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.module.auto_correlation import AutoCorrelation, AutoCorrelationLayer
from models.module.autoformer_encdec import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    my_Layernorm,
    series_decomp,
)
from models.module.embed import DataEmbedding_wo_pos


class AutoformerModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        label_len: int,
        pred_len: int,
        enc_in: int,
        dec_in: int,
        c_out: int,
        d_model: int,
        n_heads: int,
        e_layers: int,
        d_layers: int,
        d_ff: int,
        moving_avg: int,
        factor: int,
        freq: str,
        dropout: float,
        embed: str,
        num_class: int = 7,
        activation: str = "gelu",
        task_name: str = "long_term_forecast",
    ):
        super().__init__()
        self.task_name = task_name
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        kernel_size = moving_avg
        self.decomp = series_decomp(kernel_size)

        self.enc_embedding = DataEmbedding_wo_pos(enc_in, d_model, embed, freq, dropout)
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(
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
                    moving_avg=moving_avg,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=my_Layernorm(d_model),
        )

        if self.task_name in {"long_term_forecast", "short_term_forecast"}:
            self.dec_embedding = DataEmbedding_wo_pos(
                dec_in,
                d_model,
                embed,
                freq,
                dropout,
            )
            self.decoder = Decoder(
                [
                    DecoderLayer(
                        AutoCorrelationLayer(
                            AutoCorrelation(
                                True,
                                factor,
                                attention_dropout=dropout,
                                output_attention=False,
                            ),
                            d_model,
                            n_heads,
                        ),
                        AutoCorrelationLayer(
                            AutoCorrelation(
                                False,
                                factor,
                                attention_dropout=dropout,
                                output_attention=False,
                            ),
                            d_model,
                            n_heads,
                        ),
                        d_model,
                        c_out,
                        d_ff,
                        moving_avg=moving_avg,
                        dropout=dropout,
                        activation=activation,
                    )
                    for _ in range(d_layers)
                ],
                norm_layer=my_Layernorm(d_model),
                projection=nn.Linear(d_model, c_out, bias=True),
            )
        if self.task_name == "imputation":
            self.projection = nn.Linear(d_model, c_out, bias=True)
        if self.task_name == "anomaly_detection":
            self.projection = nn.Linear(d_model, c_out, bias=True)
        if self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(dropout)
            self.projection = nn.Linear(d_model * seq_len, num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros(
            [x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device
        )
        seasonal_init, trend_init = self.decomp(x_enc)
        if self.label_len == 0:
            trend_init = mean
            seasonal_init = zeros
        else:
            trend_init = torch.cat([trend_init[:, -self.label_len :, :], mean], dim=1)
            seasonal_init = torch.cat(
                [seasonal_init[:, -self.label_len :, :], zeros], dim=1
            )

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init
        )
        dec_out = trend_part + seasonal_part
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in {"long_term_forecast", "short_term_forecast"}:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len :, :]
        if self.task_name == "imputation":
            return self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
        if self.task_name == "anomaly_detection":
            return self.anomaly_detection(x_enc)
        if self.task_name == "classification":
            return self.classification(x_enc, x_mark_enc)
        return None


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        label_len: int,
        pred_len: int,
        enc_in: int,
        dec_in: int,
        c_out: int,
        d_model: int,
        n_heads: int,
        e_layers: int,
        d_layers: int,
        d_ff: int,
        moving_avg: int,
        factor: int,
        freq: str,
        dropout: float,
        embed: str,
        num_class: int,
        activation: str,
        task_name: str,
    ):
        super().__init__()
        self.model = AutoformerModel(
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            enc_in=enc_in,
            dec_in=dec_in,
            c_out=c_out,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_layers=d_layers,
            d_ff=d_ff,
            moving_avg=moving_avg,
            factor=factor,
            freq=freq,
            dropout=dropout,
            embed=embed,
            num_class=num_class,
            activation=activation,
            task_name=task_name,
        )

    def forward(self, *x):
        return self.model(*x)
