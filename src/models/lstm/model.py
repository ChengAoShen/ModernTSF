"""ModernTSF adapter for the LSTM spatiotemporal baseline from CauAir."""

from __future__ import annotations

import torch
import torch.nn as nn

from models._external.marks import to_spatiotemporal
from models.lstm._upstream import LSTM


class Model(nn.Module):
    """Adapter wrapping the CauAir LSTM baseline."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        init_dim: int = 32,
        hid_dim: int = 64,
        end_dim: int = 128,
        layer: int = 2,
        dropout: float = 0.1,
        cov_dim: int = 2,
    ) -> None:
        super().__init__()
        self.net = LSTM(
            input_dim=1 + cov_dim,
            node_num=enc_in,
            seq_len=seq_len,
            horizon=pred_len,
            init_dim=init_dim,
            hid_dim=hid_dim,
            end_dim=end_dim,
            layer=layer,
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
        if x_mark_enc is None:
            x_mark_enc = x_enc.new_zeros((x_enc.shape[0], x_enc.shape[1], 6))
        st_input = to_spatiotemporal(x_enc, x_mark_enc)  # (B, T, N, 1+F)
        out = self.net(st_input)  # (B, horizon, N, 1)
        return out.squeeze(-1)
