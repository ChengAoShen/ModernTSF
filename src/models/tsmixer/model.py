"""TSMixer model implementation."""

from __future__ import annotations

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, seq_len: int, enc_in: int, d_model: int, dropout: float) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.ReLU(),
            nn.Linear(d_model, seq_len),
            nn.Dropout(dropout),
        )
        self.channel = nn.Sequential(
            nn.Linear(enc_in, d_model),
            nn.ReLU(),
            nn.Linear(d_model, enc_in),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel(x)
        return x


class TSMixerModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int,
        e_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.layer = e_layers
        self.model = nn.ModuleList(
            [ResBlock(seq_len, enc_in, d_model, dropout) for _ in range(e_layers)]
        )
        self.pred_len = pred_len
        self.projection = nn.Linear(seq_len, pred_len)

    def forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        for i in range(self.layer):
            x_enc = self.model[i](x_enc)
        enc_out = self.projection(x_enc.transpose(1, 2)).transpose(1, 2)
        return enc_out

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del x_mark_enc, x_dec, x_mark_dec, mask
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len :, :]


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int,
        e_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.model = TSMixerModel(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            d_model=d_model,
            e_layers=e_layers,
            dropout=dropout,
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
