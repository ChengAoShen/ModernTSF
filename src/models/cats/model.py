"""ModernTSF adapter for the CATS model."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.cats._upstream import CATS as _CATS


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 128,
        n_heads: int = 16,
        d_ff: int = 256,
        n_layers: int = 3,
        dropout: float = 0.1,
        stride: int = 24,
        QAM_start: float = 0.1,
        QAM_end: float = 0.5,
    ) -> None:
        super().__init__()
        self.net = _CATS(
            node_num=enc_in,
            seq_len=seq_len,
            horizon=pred_len,
            stride=stride,
            n_layers=n_layers,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            QAM_start=QAM_start,
            QAM_end=QAM_end,
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x_enc: (B, T, N)
        # CATS expects (B, T, N, F) and uses z[..., 0].transpose(1,-1)
        x = x_enc.unsqueeze(-1)  # (B, T, N, 1)
        out = self.net(x)  # (B, horizon, N, 1)
        return out.squeeze(-1)  # (B, pred_len, N)
