"""ModernTSF adapter for the CauAir air-quality forecasting model.

CauAir (https://github.com/PoorOtterBob/CauAir) forecasts a target series
from its history plus a block of *future* covariates. It consumes:

* ``x``     : ``(B, T, N, input_dim)`` history; channel 0 is the value,
              channels ``1:`` are covariates.
* ``label`` : ``(B, T, N, input_dim - 1)`` future covariates.

and returns ``(B, horizon, N, 1)``.

In the generic benchmark the only covariates available are the normalized
calendar features, so ``input_dim = 1 + F``. The future covariate block is
the normalized decoder marks, broadcast across nodes and coerced to the
``seq_len`` the upstream reshape expects.
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
from models.cauair._upstream import CauAir


class Model(nn.Module):
    """Adapter wrapping the upstream CauAir model.

    Parameters
    ----------
    seq_len : int
        Input sequence length (also the future-covariate block length).
    pred_len : int
        Forecast horizon.
    enc_in : int
        Number of spatial nodes (channels).
    cov_dim : int, optional
        Number of covariate channels ``F`` per node. Defaults to the calendar
        feature count (time-of-day, day-of-week). In spatiotemporal /
        air-quality mode set this to the dataset's covariate count so the
        future-covariate block is sized correctly.
    dim : int
        Hidden dimension of the encoders / decoder.
    rank : int
        Rank of the CachLormer low-rank attention.
    head : int
        Number of attention heads.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        cov_dim: int | None = None,
        dim: int = 64,
        rank: int = 8,
        head: int = 4,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        cov = TIME_FEATURES if cov_dim is None else cov_dim
        self.input_dim = 1 + cov
        self.net = CauAir(
            dim=dim,
            rank=rank,
            head=head,
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

        Parameters
        ----------
        x_enc : torch.Tensor
            Input values of shape ``(B, seq_len, N)``.
        x_mark_enc : torch.Tensor, optional
            Raw input marks of shape ``(B, seq_len, 6)``.
        x_dec : torch.Tensor, optional
            Unused (CauAir uses future covariates, not future values).
        x_mark_dec : torch.Tensor, optional
            Raw future marks of shape ``(B, label_len + pred_len, 6)``.
        mask : torch.Tensor, optional
            Unused.

        Returns
        -------
        torch.Tensor
            Forecast of shape ``(B, pred_len, N)``.
        """
        b, t, n = x_enc.shape
        if x_mark_enc is None:
            x_mark_enc = x_enc.new_zeros((b, t, 6))
        history = to_spatiotemporal(x_enc, x_mark_enc)  # (B, T, N, 1 + F)

        if x_mark_dec is None:
            future_marks = x_mark_enc
        else:
            future_marks = x_mark_dec
        future_marks = coerce_time_length(future_marks, self.seq_len)
        future = future_time_features(future_marks, n)  # (B, seq_len, N, F)

        out = self.net(history, future)  # (B, horizon, N, 1)
        return out.squeeze(-1)
