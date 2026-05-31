"""ModernTSF adapter for the HL (Historical Last) baseline.

HL simply repeats the last observed value across the forecast horizon.
It serves as a naive lower-bound baseline for spatiotemporal forecasting.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models._external.marks import to_spatiotemporal
from models.hl._upstream import HL


class Model(nn.Module):
    """Adapter wrapping the HL baseline."""

    def __init__(self, seq_len: int, pred_len: int, enc_in: int) -> None:
        super().__init__()
        self.net = HL(horizon=pred_len)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_mark_enc is None:
            x_mark_enc = x_enc.new_zeros((x_enc.shape[0], x_enc.shape[1], 6))
        st_input = to_spatiotemporal(x_enc, x_mark_enc)  # (B, T, N, 1+F)
        out = self.net(st_input)  # (B, horizon, N, 1)
        return out.squeeze(-1)
