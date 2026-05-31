"""ModernTSF adapter for the PCDCNet spatiotemporal forecasting model.

PCDCNet uses GCN + GRU with physics-constrained causal structure.
It consumes history ``(B, T, N, F)`` and future covariates ``(B, T, N, F-1)``
and returns ``(B, horizon, N, output_dim)`` squeezed to ``(B, pred_len, N)``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models._external.marks import (
    TIME_FEATURES,
    coerce_time_length,
    future_time_features,
    to_spatiotemporal,
)
from models.pcdcnet._upstream import PCDCNet


class Model(nn.Module):
    """Adapter wrapping the upstream PCDCNet model."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        adj_mx: np.ndarray | None = None,
        cov_dim: int | None = None,
        d_model: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if adj_mx is None:
            adj_mx = np.eye(enc_in, dtype=np.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        cov = TIME_FEATURES if cov_dim is None else cov_dim
        self.input_dim = 1 + cov
        self.net = PCDCNet(
            adj_mx=adj_mx,
            node_num=enc_in,
            input_dim=self.input_dim,
            output_dim=1,
            seq_len=seq_len,
            horizon=pred_len,
            d_model=d_model,
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
        b, t, n = x_enc.shape
        if x_mark_enc is None:
            x_mark_enc = x_enc.new_zeros((b, t, 6))
        history = to_spatiotemporal(x_enc, x_mark_enc)  # (B, T, N, 1+F)

        if x_mark_dec is None:
            future_marks = x_mark_enc
        else:
            future_marks = x_mark_dec
        future_marks = coerce_time_length(future_marks, self.seq_len)
        future = future_time_features(future_marks, n)  # (B, seq_len, N, F)

        out = self.net(history, future)  # (B, horizon, N, 1)
        out = out.squeeze(-1)
        return out[:, :self.pred_len, :]
