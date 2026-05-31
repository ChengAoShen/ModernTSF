"""ModernTSF adapter for the UMixer model."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.umixer._upstream import UMixer as _UMixer


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        patch_len: int = 24,
        stride: int = 24,
        d_model: int = 128,
        dropout: float = 0.1,
        e_layers: int = 2,
        d_layers: int = 1,
    ) -> None:
        super().__init__()
        self.net = _UMixer(
            node_num=enc_in,
            seq_len=seq_len,
            horizon=pred_len,
            stride=stride,
            patch_len=patch_len,
            d_model=d_model,
            dropout=dropout,
            e_layers=e_layers,
            d_layers=d_layers,
            enc_in=enc_in,
            c_out=enc_in,
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
        out = self.net.forecast(x_enc)
        return out  # (B, pred_len, N)
