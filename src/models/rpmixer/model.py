"""ModernTSF adapter for the RPMixer forecasting model.

RPMixer consumes:
* ``x``     : ``(B, T, N, input_dim)`` history; channel 0 is the value,
              channels ``1:`` are covariates.
* ``label`` : ``(B, T, N, input_dim - 1)`` future covariates.

and returns ``(B, horizon, N, 1)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models._external.marks import (
    TIME_FEATURES,
    coerce_time_length,
    future_time_features,
    to_spatiotemporal,
)
from models.rpmixer._upstream import RPMixer


class Model(nn.Module):
    """Adapter wrapping the upstream RPMixer model."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        cov_dim: int | None = None,
        IE_dim: int = 32,
        dropout: float = 0.3,
        num_head: int = 2,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        cov = TIME_FEATURES if cov_dim is None else cov_dim
        self.input_dim = 1 + cov
        self.net = RPMixer(
            node_num=enc_in,
            input_dim=self.input_dim,
            output_dim=1,
            seq_len=seq_len,
            horizon=pred_len,
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
        return out.squeeze(-1)
