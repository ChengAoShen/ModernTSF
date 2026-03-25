"""MixLinear model implementation."""

from __future__ import annotations

import torch
import torch.nn as nn


class MixLinearModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        period_len: int,
        com_len: int,
        lpf: int,
        alpha: float,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.period_len = period_len
        self.com_len = com_len
        self.lpf = lpf
        self.alpha = alpha

        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len

        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=1 + 2 * self.period_len // 2,
            stride=1,
            padding=self.period_len // 2,
            padding_mode="zeros",
            bias=False,
        )

        self.linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)
        self.linear1 = nn.Linear(self.seg_num_x, self.com_len, bias=False)
        self.linear2 = nn.Linear(self.com_len, self.seg_num_y, bias=False)

        self.flinear1 = nn.Linear(self.lpf, 2, bias=False).to(torch.cfloat)
        self.flinear2 = nn.Linear(2, self.seg_num_y, bias=False).to(torch.cfloat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = (x - seq_mean).permute(0, 2, 1)

        x = (
            self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(
                -1, self.enc_in, self.seq_len
            )
            + x
        )

        if self.seq_len % self.period_len != 0:
            x = x[:, :, -(self.seg_num_x * self.period_len) :]

        x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)

        y_t = self.linear2(self.linear1(x))

        x_fft = torch.fft.fft(x, dim=2)
        if x_fft.size(-1) < self.lpf:
            pad_size = self.lpf - x_fft.size(-1)
            pad = x_fft.new_zeros((*x_fft.shape[:-1], pad_size))
            x_fft = torch.cat([x_fft, pad], dim=-1)
        else:
            x_fft = x_fft[:, :, : self.lpf]
        x_fft = self.flinear2(self.flinear1(x_fft))

        y_f = torch.fft.ifft(x_fft, dim=2).float()

        y = y_t * self.alpha + y_f * (1 - self.alpha)

        y = y.permute(0, 2, 1).reshape(batch_size, self.enc_in, -1)
        y = y[:, :, : self.pred_len]
        y = y.permute(0, 2, 1) + seq_mean
        return y


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        period_len: int,
        com_len: int,
        lpf: int,
        alpha: float,
    ):
        super().__init__()
        self.model = MixLinearModel(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            period_len=period_len,
            com_len=com_len,
            lpf=lpf,
            alpha=alpha,
        )

    def forward(self, x, *args):
        return self.model(x)
