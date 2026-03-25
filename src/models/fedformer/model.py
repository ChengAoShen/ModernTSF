"""FEDformer model implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.module.auto_correlation import AutoCorrelationLayer
from models.module.autoformer_encdec import (
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderLayer,
    my_Layernorm,
    series_decomp,
)
from models.module.embed import DataEmbedding
from models.module.fourier_correlation import FourierBlock, FourierCrossAttention


class FEDformerModel(nn.Module):
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
        freq: str,
        dropout: float,
        embed: str,
        num_class: int = 7,
        activation: str = "gelu",
        task_name: str = "long_term_forecast",
        version: str = "fourier",
        mode_select: str = "random",
        modes: int = 32,
    ):
        super().__init__()
        self.task_name = task_name
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len

        self.version = version
        self.mode_select = mode_select
        self.modes = modes

        self.decomp = series_decomp(moving_avg)
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.dec_embedding = DataEmbedding(dec_in, d_model, embed, freq, dropout)

        if self.version.lower() == "wavelets":
            from models.module.multi_wavelet_correlation import (
                MultiWaveletCross,
                MultiWaveletTransform,
            )

            encoder_self_att = MultiWaveletTransform(ich=d_model, L=1, base="legendre")
            decoder_self_att = MultiWaveletTransform(ich=d_model, L=1, base="legendre")
            decoder_cross_att = MultiWaveletCross(
                in_channels=d_model,
                out_channels=d_model,
                seq_len_q=self.seq_len // 2 + self.pred_len,
                seq_len_kv=self.seq_len,
                modes=self.modes,
                ich=d_model,
                base="legendre",
                activation="tanh",
            )
        else:
            encoder_self_att = FourierBlock(
                in_channels=d_model,
                out_channels=d_model,
                n_heads=n_heads,
                seq_len=self.seq_len,
                modes=self.modes,
                mode_select_method=self.mode_select,
            )
            decoder_self_att = FourierBlock(
                in_channels=d_model,
                out_channels=d_model,
                n_heads=n_heads,
                seq_len=self.seq_len // 2 + self.pred_len,
                modes=self.modes,
                mode_select_method=self.mode_select,
            )
            decoder_cross_att = FourierCrossAttention(
                in_channels=d_model,
                out_channels=d_model,
                seq_len_q=self.seq_len // 2 + self.pred_len,
                seq_len_kv=self.seq_len,
                modes=self.modes,
                mode_select_method=self.mode_select,
                num_heads=n_heads,
            )

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(encoder_self_att, d_model, n_heads),
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
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(decoder_self_att, d_model, n_heads),
                    AutoCorrelationLayer(decoder_cross_att, d_model, n_heads),
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
        seasonal_init, trend_init = self.decomp(x_enc)
        if self.label_len == 0:
            trend_init = mean
            seasonal_init = F.pad(seasonal_init[:, :0, :], (0, 0, 0, self.pred_len))
        else:
            trend_init = torch.cat([trend_init[:, -self.label_len :, :], mean], dim=1)
            seasonal_init = F.pad(
                seasonal_init[:, -self.label_len :, :], (0, 0, 0, self.pred_len)
            )

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)
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
        freq: str,
        dropout: float,
        embed: str,
        num_class: int,
        activation: str,
        task_name: str,
        version: str,
        mode_select: str,
        modes: int,
    ):
        super().__init__()
        self.model = FEDformerModel(
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
            freq=freq,
            dropout=dropout,
            embed=embed,
            num_class=num_class,
            activation=activation,
            task_name=task_name,
            version=version,
            mode_select=mode_select,
            modes=modes,
        )

    def forward(self, *x):
        return self.model(*x)
