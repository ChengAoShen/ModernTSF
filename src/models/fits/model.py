"""FITS model implementation."""

from __future__ import annotations

import torch
import torch.nn as nn


class FITSModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        individual: bool,
        cut_freq: int,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.individual = individual
        self.channels = enc_in
        self.dominance_freq = cut_freq
        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len

        if self.individual:
            self.freq_upsampler = nn.ModuleList()
            for _ in range(self.channels):
                self.freq_upsampler.append(
                    nn.Linear(
                        self.dominance_freq,
                        int(self.dominance_freq * self.length_ratio),
                    ).to(torch.cfloat)
                )
        else:
            self.freq_upsampler = nn.Linear(
                self.dominance_freq, int(self.dominance_freq * self.length_ratio)
            ).to(torch.cfloat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        x = x / torch.sqrt(x_var)

        low_specx = torch.fft.rfft(x, dim=1)
        low_specx[:, self.dominance_freq :] = 0
        low_specx = low_specx[:, 0 : self.dominance_freq, :]

        if self.individual:
            low_specxy_ = torch.zeros(
                [
                    low_specx.size(0),
                    int(self.dominance_freq * self.length_ratio),
                    low_specx.size(2),
                ],
                dtype=low_specx.dtype,
                device=low_specx.device,
            )
            for i in range(self.channels):
                low_specxy_[:, :, i] = self.freq_upsampler[i](
                    low_specx[:, :, i].permute(0, 1)
                ).permute(0, 1)
        else:
            low_specxy_ = self.freq_upsampler(low_specx.permute(0, 2, 1)).permute(
                0, 2, 1
            )

        low_specxy = torch.zeros(
            [
                low_specxy_.size(0),
                int((self.seq_len + self.pred_len) / 2 + 1),
                low_specxy_.size(2),
            ],
            dtype=low_specxy_.dtype,
            device=low_specxy_.device,
        )
        low_specxy[:, 0 : low_specxy_.size(1), :] = low_specxy_
        low_xy = torch.fft.irfft(low_specxy, dim=1)
        low_xy = low_xy * self.length_ratio

        xy = low_xy * torch.sqrt(x_var) + x_mean
        return xy


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        individual: bool,
        cut_freq: int,
    ):
        super().__init__()
        self.model = FITSModel(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            individual=individual,
            cut_freq=cut_freq,
        )

    def forward(self, x, *args):
        return self.model(x)
