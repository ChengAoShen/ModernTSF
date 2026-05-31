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
    ):
        super().__init__()
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
        self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.projection = nn.Linear(d_model, c_out, bias=True)

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

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len :, :]


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
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.model(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
