"""ModernTSF adapter for the STID spatiotemporal forecasting model.

STID uses spatial-temporal identity embeddings for forecasting.
It consumes ``(B, T, N, F)`` and returns ``(B, horizon, N, 1)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models._external.marks import to_spatiotemporal
from models.stid._upstream import STID


class Model(nn.Module):
    """Adapter wrapping the upstream STID model."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        cov_dim: int = 2,
        hid_dim: int = 64,
        num_layers: int = 3,
        time_of_day_size: int = 24,
        day_of_week_size: int = 7,
    ) -> None:
        super().__init__()
        input_dim = 1 + cov_dim
        self.net = STID(
            tod=time_of_day_size,
            node_num=enc_in,
            input_dim=input_dim,
            output_dim=1,
            seq_len=seq_len,
            horizon=pred_len,
        )
        self.pred_len = pred_len

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_mark_enc is None:
            x_mark_enc = x_enc.new_zeros(
                (x_enc.shape[0], x_enc.shape[1], 6))
        st_input = to_spatiotemporal(x_enc, x_mark_enc)
        # STID output: (B, output_len, N, 1)
        out = self.net(st_input)
        # Squeeze trailing dim
        if out.dim() == 4:
            out = out.squeeze(-1)
        return out[:, :self.pred_len, :]
