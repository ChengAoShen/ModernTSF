"""PaiFilter model implementation."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.module.revin import RevIN


class PaiFilterModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        hidden_size: int,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scale = 0.02
        self.revin_layer = RevIN(enc_in, affine=True, subtract_last=False)

        self.embed_size = self.seq_len
        self.hidden_size = hidden_size

        self.w = nn.Parameter(self.scale * torch.randn(1, self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len),
        )

    def circular_convolution(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        x = torch.fft.rfft(x, dim=2, norm="ortho")
        w = torch.fft.rfft(w, dim=1, norm="ortho")
        y = x * w
        out = torch.fft.irfft(y, n=self.embed_size, dim=2, norm="ortho")
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.revin_layer(x, "norm")

        x = z.permute(0, 2, 1)
        x = self.circular_convolution(x, self.w.to(x.device))
        x = self.fc(x)
        x = x.permute(0, 2, 1)

        z = self.revin_layer(x, "denorm")
        return z


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        hidden_size: int,
    ):
        super().__init__()
        self.model = PaiFilterModel(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            hidden_size=hidden_size,
        )

    def forward(self, x, *args):
        return self.model(x)
