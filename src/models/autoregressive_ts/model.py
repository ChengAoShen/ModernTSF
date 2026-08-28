"""Independent direct multi-horizon autoregressive baseline."""

from __future__ import annotations

import torch
import torch.nn as nn


class Model(nn.Module):
    """Apply one shared lag regression to every channel independently."""

    def __init__(self, seq_len: int, pred_len: int, enc_in: int) -> None:
        super().__init__()
        if seq_len < 1 or pred_len < 1 or enc_in < 1:
            raise ValueError("seq_len, pred_len, and enc_in must be positive")
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.projection = nn.Linear(seq_len, pred_len)
        self.aux_loss: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, *args: object) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1:] != (self.seq_len, self.enc_in):
            raise ValueError(
                f"expected [batch, {self.seq_len}, {self.enc_in}], got {tuple(x.shape)}"
            )
        forecast = self.projection(x.transpose(1, 2)).transpose(1, 2)
        self.aux_loss = forecast.new_zeros(())
        return forecast
