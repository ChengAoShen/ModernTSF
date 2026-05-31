"""ModernTSF adapter for the TimeXer model."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.timexer._upstream import TimeXer as _TimeXer


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        patch_len: int = 24,
        d_model: int = 128,
        dropout: float = 0.1,
        n_heads: int = 4,
        e_layers: int = 2,
        d_ff: int = 256,
    ) -> None:
        super().__init__()
        self.net = _TimeXer(
            node_num=enc_in,
            seq_len=seq_len,
            horizon=pred_len,
            input_dim=1,
            patch_len=patch_len,
            d_model=d_model,
            dropout=dropout,
            factor=1,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            features='M',
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x_enc: (B, T, N) — nodes as channels
        # Use forecast_multi for multivariate mode; pass None for x_mark_enc
        # so DataEmbedding_inverted uses only x
        out = self.net.forecast_multi(x_enc, None, x_dec, x_mark_dec)
        return out[:, -self.net.pred_len:, :]  # (B, pred_len, N)
