"""ModernTSF adapter for the DSFormer model."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.dsformer._upstream import DSFormer as _DSFormer


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        num_layer: int = 1,
        dropout: float = 0.2,
        muti_head: int = 4,
        num_samp: int = 3,
    ) -> None:
        super().__init__()
        self.net = _DSFormer(
            node_num=enc_in,
            seq_len=seq_len,
            horizon=pred_len,
            num_layer=num_layer,
            dropout=dropout,
            muti_head=muti_head,
            num_samp=num_samp,
            IF_node=True,
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x_enc: (B, T, N) — pass directly, DSFormer expects (B, H, N)
        out = self.net(x_enc)  # (B, L, N)
        return out  # (B, pred_len, N)
