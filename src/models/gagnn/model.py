"""ModernTSF adapter for the GAGNN spatiotemporal forecasting model.

GAGNN uses group-aware graph neural networks for air quality forecasting.
It consumes ``(B, T, N, F)`` and returns ``(B, horizon, N, output_dim)``
which is squeezed to ``(B, pred_len, N)``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models._external.marks import to_spatiotemporal
from models.gagnn._upstream import GAGNN


class Model(nn.Module):
    """Adapter wrapping the upstream GAGNN model."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        adj_mx: np.ndarray | None = None,
        cov_dim: int = 2,
        d_model: int = 64,
        n_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if adj_mx is None:
            adj_mx = np.eye(enc_in, dtype=np.float32)
        input_dim = 1 + cov_dim
        self.pred_len = pred_len
        self.net = GAGNN(
            node_num=enc_in,
            input_dim=input_dim,
            output_dim=1,
            seq_len=seq_len,
            horizon=pred_len,
            adj_mx=adj_mx,
            d_model=d_model,
            n_heads=n_heads,
            num_layers=num_layers,
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
        """Forecast future values.

        Returns
        -------
        torch.Tensor
            Forecast of shape ``(B, pred_len, N)``.
        """
        if x_mark_enc is None:
            x_mark_enc = x_enc.new_zeros(
                (x_enc.shape[0], x_enc.shape[1], 6))
        st_input = to_spatiotemporal(x_enc, x_mark_enc)
        # out: (B, horizon, N, output_dim)
        out = self.net(st_input)
        # squeeze output_dim=1
        out = out.squeeze(-1)  # (B, horizon, N)
        return out[:, :self.pred_len, :]
