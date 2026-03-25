"""Amplifier model implementation."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.amplifier.layers import SeriesDecomp
from models.module.revin import RevIN


class AmplifierModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        hidden_size: int,
        sci: bool,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.hidden_size = hidden_size
        self.revin_layer = RevIN(enc_in, affine=True, subtract_last=False)

        kernel_size = 25
        self.decomposition = SeriesDecomp(kernel_size)

        self.mask_matrix = nn.Parameter(
            torch.ones(int(self.seq_len / 2) + 1, self.channels)
        )
        self.freq_linear = nn.Linear(
            int(self.seq_len / 2) + 1, int(self.pred_len / 2) + 1
        ).to(torch.cfloat)

        self.linear_seasonal = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len),
        )
        self.linear_trend = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len),
        )

        self.sci = sci
        self.extract_common_pattern = nn.Sequential(
            nn.Linear(self.channels, self.channels),
            nn.LeakyReLU(),
            nn.Linear(self.channels, 1),
        )
        self.model_common_pattern = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.seq_len),
        )
        self.model_specific_pattern = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.seq_len),
        )

    def forward(
        self,
        x: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del x_mark_enc, x_dec, x_mark_dec, mask
        batch_size, _, channels = x.size()

        z = self.revin_layer(x, "norm")
        x = z

        x_fft = torch.fft.rfft(x, dim=1)
        x_inverse_fft = torch.flip(x_fft, dims=[1])
        x_inverse_fft = x_inverse_fft * self.mask_matrix
        x_amplifier_fft = x_fft + x_inverse_fft
        x_amplifier = torch.fft.irfft(x_amplifier_fft, dim=1)

        if self.sci:
            x = x_amplifier
            common_pattern = self.extract_common_pattern(x)
            common_pattern = self.model_common_pattern(
                common_pattern.permute(0, 2, 1)
            ).permute(0, 2, 1)
            specific_pattern = x - common_pattern.repeat(1, 1, channels)
            specific_pattern = self.model_specific_pattern(
                specific_pattern.permute(0, 2, 1)
            ).permute(0, 2, 1)
            x_amplifier = specific_pattern + common_pattern.repeat(1, 1, channels)

        seasonal, trend = self.decomposition(x_amplifier)
        seasonal = self.linear_seasonal(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
        trend = self.linear_trend(trend.permute(0, 2, 1)).permute(0, 2, 1)
        out_amplifier = seasonal + trend

        out_amplifier_fft = torch.fft.rfft(out_amplifier, dim=1)
        x_inverse_fft = self.freq_linear(x_inverse_fft.permute(0, 2, 1)).permute(
            0, 2, 1
        )
        out_fft = out_amplifier_fft - x_inverse_fft
        out = torch.fft.irfft(out_fft, dim=1)

        z = self.revin_layer(out, "denorm")
        return z


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        hidden_size: int,
        sci: bool,
    ) -> None:
        super().__init__()
        self.model = AmplifierModel(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            hidden_size=hidden_size,
            sci=sci,
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
