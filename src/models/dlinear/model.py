"""Paper-driven local implementation of decomposition-linear forecasting."""

from __future__ import annotations

import torch
from torch import nn

from components.dlinear import DLinearBackbone


class Model(nn.Module):
    """Sum linear forecasts of moving-average trend and seasonal remainder."""

    def __init__(self, c_in: int, seq_len: int, pred_len: int,
                 kernel_size: int = 25, individual: bool = False) -> None:
        super().__init__()
        if min(c_in, seq_len, pred_len) < 1:
            raise ValueError("channels and sequence lengths must be positive")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.c_in = c_in
        self.seq_len = seq_len
        self.backbone = DLinearBackbone(
            c_in, seq_len, pred_len, kernel_size=kernel_size, individual=individual
        )

    def forward(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor | None = None,
                x_dec: torch.Tensor | None = None, x_mark_dec: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        del x_mark_enc, x_dec, x_mark_dec, mask
        if x_enc.shape[1:] != (self.seq_len, self.c_in):
            raise ValueError("x_enc does not match configured time/channel dimensions")
        return self.backbone(x_enc)
