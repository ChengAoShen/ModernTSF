"""TimeEmb model implementation."""

from __future__ import annotations

import torch
import torch.nn as nn


class TimeEmbModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int,
        use_revin: bool,
        use_hour_index: bool,
        use_day_index: bool,
        scale: float,
        emb_len_hour: int,
        emb_len_day: int,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.d_model = d_model
        self.use_revin = use_revin
        self.use_day_index = use_day_index
        self.use_hour_index = use_hour_index
        self.scale = scale
        self.emb_len_hour = emb_len_hour
        self.emb_len_day = emb_len_day

        self.model = nn.Sequential(
            nn.Linear(self.seq_len, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.pred_len),
        )

        self.emb_hour = nn.Parameter(
            torch.zeros(self.emb_len_hour, self.enc_in, self.seq_len // 2 + 1),
            requires_grad=True,
        )
        self.emb_day = nn.Parameter(
            torch.zeros(self.emb_len_day, self.enc_in, self.seq_len // 2 + 1),
            requires_grad=True,
        )
        self.w = nn.Parameter(self.scale * torch.randn(1, self.seq_len))

    def forward(
        self, x: torch.Tensor, hour_index: torch.Tensor, day_index: torch.Tensor | None
    ) -> torch.Tensor:
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        x = x.permute(0, 2, 1)
        x = torch.fft.rfft(x, dim=2, norm="ortho")
        w = torch.fft.rfft(self.w, dim=1, norm="ortho")
        x_freq_real = x.real
        x_freq_imag = x.imag

        if self.use_hour_index:
            emb_hour = self.emb_hour[hour_index % self.emb_len_hour]
            x_freq_real = x_freq_real - emb_hour

        if self.use_day_index and day_index is not None:
            emb_day = self.emb_day[day_index % self.emb_len_day]
            x_freq_real = x_freq_real - emb_day

        x_freq_minus_emb = torch.complex(x_freq_real, x_freq_imag)
        y = x_freq_minus_emb * w
        y_real = y.real
        y_freq_imag = y.imag

        if self.use_day_index and day_index is not None:
            y_real = y_real + emb_day
        if self.use_hour_index:
            y_real = y_real + emb_hour

        y_freq = torch.complex(y_real, y_freq_imag)
        y = torch.fft.irfft(y_freq, n=self.seq_len, dim=2, norm="ortho")
        y = self.model(y).permute(0, 2, 1)

        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean

        return y


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int,
        use_revin: bool,
        use_hour_index: bool,
        use_day_index: bool,
        scale: float,
        hour_length: int,
        day_length: int,
    ):
        super().__init__()
        self.model = TimeEmbModel(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            d_model=d_model,
            use_revin=use_revin,
            use_hour_index=use_hour_index,
            use_day_index=use_day_index,
            scale=scale,
            emb_len_hour=hour_length,
            emb_len_day=day_length,
        )

    def forward(self, x, x_time_stamp, *args):
        day_index = x_time_stamp[:, -1, 2].to(torch.int64)
        hour_index = x_time_stamp[:, -1, 3].to(torch.int64)
        return self.model(x, hour_index, day_index)
