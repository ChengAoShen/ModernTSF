"""TimesNet model implementation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.module.conv_blocks import Inception_Block_V1
from models.module.embed import DataEmbedding


def FFT_for_Period(x, k=2):
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, seq_len, pred_len, top_k, d_model, d_ff, num_kernels=3):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels),
        )

    def forward(self, x):
        batch_size, input_length, channels = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]
            if (self.seq_len + self.pred_len) % period != 0:
                length = (((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros(
                    [x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]
                ).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.seq_len + self.pred_len
                out = x
            out = (
                out.reshape(batch_size, length // period, period, channels)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            out = self.conv(out)
            out = out.permute(0, 2, 3, 1).reshape(batch_size, -1, channels)
            res.append(out[:, : (self.seq_len + self.pred_len), :])
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = (
            period_weight.unsqueeze(1).unsqueeze(1).repeat(1, input_length, channels, 1)
        )
        res = torch.sum(res * period_weight, -1)
        res = res + x
        return res


class TimesNetModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        label_len: int,
        pred_len: int,
        enc_in: int,
        c_out: int,
        d_model: int,
        e_layers: int,
        d_ff: int,
        freq: str,
        dropout: float,
        embed: str,
        top_k: int = 3,
        num_kernels: int = 3,
        num_class: int = 7,
        task_name: str = "long_term_forecast",
    ):
        super().__init__()
        self.task_name = task_name
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.model = nn.ModuleList(
            [
                TimesBlock(seq_len, pred_len, top_k, d_model, d_ff, num_kernels)
                for _ in range(e_layers)
            ]
        )
        self.enc_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.layer = e_layers
        self.layer_norm = nn.LayerNorm(d_model)
        if self.task_name in ("long_term_forecast", "short_term_forecast"):
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.projection = nn.Linear(d_model, c_out, bias=True)
        if self.task_name in ("imputation", "anomaly_detection"):
            self.projection = nn.Linear(d_model, c_out, bias=True)
        if self.task_name == "classification":
            self.act = F.gelu
            self.dropout = nn.Dropout(dropout)
            self.projection = nn.Linear(d_model * seq_len, num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(0, 2, 1)
        for layer in range(self.layer):
            enc_out = self.layer_norm(self.model[layer](enc_out))
        dec_out = self.projection(enc_out)

        dec_out = dec_out.mul(
            (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        )
        dec_out = dec_out.add(
            (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        )
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc.sub(means)
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(
            torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5
        )
        stdev = stdev.unsqueeze(1).detach()
        x_enc = x_enc.div(stdev)

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        for layer in range(self.layer):
            enc_out = self.layer_norm(self.model[layer](enc_out))
        dec_out = self.projection(enc_out)

        dec_out = dec_out.mul(
            (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        )
        dec_out = dec_out.add(
            (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        )
        return dec_out

    def anomaly_detection(self, x_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        enc_out = self.enc_embedding(x_enc, None)
        for layer in range(self.layer):
            enc_out = self.layer_norm(self.model[layer](enc_out))
        dec_out = self.projection(enc_out)

        dec_out = dec_out.mul(
            (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        )
        dec_out = dec_out.add(
            (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1))
        )
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, None)
        for layer in range(self.layer):
            enc_out = self.layer_norm(self.model[layer](enc_out))

        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ("long_term_forecast", "short_term_forecast"):
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
        c_out: int,
        d_model: int,
        e_layers: int,
        d_ff: int,
        freq: str,
        dropout: float,
        embed: str,
        top_k: int,
        num_kernels: int,
        num_class: int,
        task_name: str,
    ):
        super().__init__()
        self.model = TimesNetModel(
            seq_len=seq_len,
            label_len=label_len,
            pred_len=pred_len,
            enc_in=enc_in,
            c_out=c_out,
            d_model=d_model,
            e_layers=e_layers,
            d_ff=d_ff,
            freq=freq,
            dropout=dropout,
            embed=embed,
            top_k=top_k,
            num_kernels=num_kernels,
            num_class=num_class,
            task_name=task_name,
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.model(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
