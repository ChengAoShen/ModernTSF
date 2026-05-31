"""ModernTSF adapter for the MGSFformer forecasting model.

MGSFformer consumes (B, H, N, 1) using only channel 0 of the input
and outputs (B, L, N, 1).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.mgsfformer._upstream import MGSFformer


class Model(nn.Module):
    """Adapter wrapping the upstream MGSFformer model."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        IE_dim: int = 32,
        dropout: float = 0.3,
        num_head: int = 2,
    ) -> None:
        super().__init__()
        self.net = MGSFformer(
            node_num=enc_in,
            input_dim=1,
            output_dim=1,
            seq_len=seq_len,
            horizon=pred_len,
            Input_len=seq_len,
            out_len=pred_len,
            num_id=enc_in,
            IE_dim=IE_dim,
            dropout=dropout,
            num_head=num_head,
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forecast future values.

        Returns (B, pred_len, N).
        """
        # MGSFformer expects (B, H, N, 1), uses only channel 0
        history = x_enc.unsqueeze(-1)  # (B, T, N, 1)
        out = self.net(history)  # (B, L, N, 1)
        return out.squeeze(-1)  # (B, pred_len, N)
