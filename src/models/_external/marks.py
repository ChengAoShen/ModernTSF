"""Shared input-adaptation helpers for ported external models.

ModernTSF feeds models a 4-tuple ``(x_enc, x_mark_enc, x_dec, x_mark_dec)``:

* ``x_enc``      : ``(B, seq_len, N)``  value tensor (N channels / nodes)
* ``x_mark_enc`` : ``(B, seq_len, 6)``  raw integer time features
                   ``[year, month, day, weekday, hour, minute]``
* ``x_dec``      : ``(B, label_len + pred_len, N)`` decoder value input
* ``x_mark_dec`` : ``(B, label_len + pred_len, 6)`` raw decoder time features

The spatiotemporal / air-quality models ported here were trained on the
BasicTS / LargeST convention, where each node carries ``1 + F`` channels:
the measured value plus ``F`` *normalized* calendar features. Following the
user's specification we use the two coarsest-useful features (up to
day-of-week), so ``F = 2``:

* ``time_in_day`` = ``(hour * 60 + minute) / 1440`` in ``[0, 1)``
* ``day_in_week`` = ``weekday / 7``                 in ``[0, 1)``

These helpers convert the framework marks into the layout each family of
models expects, keeping every adapter thin and consistent.
"""

from __future__ import annotations

import torch


# Index positions inside the raw 6-column time-stamp produced by
# ``ForecastingDataset._build_time_stamp``.
_WEEKDAY = 3
_HOUR = 4
_MINUTE = 5

# Number of stacked calendar features (time-of-day, day-of-week).
TIME_FEATURES = 2


def normalized_time_features(marks: torch.Tensor) -> torch.Tensor:
    """Convert raw integer marks to normalized calendar features.

    Parameters
    ----------
    marks : torch.Tensor
        Raw time features of shape ``(B, T, 6)`` with columns
        ``[year, month, day, weekday, hour, minute]``.

    Returns
    -------
    torch.Tensor
        Normalized features of shape ``(B, T, 2)`` ordered as
        ``[time_in_day, day_in_week]``, both in ``[0, 1)``.
    """
    hour = marks[..., _HOUR]
    minute = marks[..., _MINUTE]
    weekday = marks[..., _WEEKDAY]

    time_in_day = (hour * 60.0 + minute) / 1440.0
    day_in_week = weekday / 7.0
    return torch.stack([time_in_day, day_in_week], dim=-1)


def to_spatiotemporal(values: torch.Tensor, marks: torch.Tensor) -> torch.Tensor:
    """Build a ``(B, T, N, 1 + F)`` spatiotemporal tensor.

    The value tensor becomes channel 0; the ``F`` covariate / calendar
    features fill the remaining channels. The covariate source depends on the
    rank of ``marks``:

    * ``marks`` is ``(B, T, N, F)`` — node-structured covariates supplied by a
      spatiotemporal / air-quality dataset; used as-is.
    * ``marks`` is ``(B, T, 6)`` — raw calendar stamps from a forecasting-style
      dataset; converted to ``[time_in_day, day_in_week]`` and broadcast across
      the ``N`` nodes.

    Parameters
    ----------
    values : torch.Tensor
        Value tensor of shape ``(B, T, N)``.
    marks : torch.Tensor
        Either node-structured covariates ``(B, T, N, F)`` or raw calendar
        stamps ``(B, T, 6)``.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(B, T, N, 1 + F)``.
    """
    b, t, n = values.shape
    value_channel = values.unsqueeze(-1)  # (B, T, N, 1)
    if marks is not None and marks.dim() == 4:
        # Already node-structured covariates (B, T, N, F).
        feats = marks
    else:
        if marks is None:
            marks = values.new_zeros((b, t, 6))
        feats = normalized_time_features(marks)  # (B, T, F)
        feats = feats.unsqueeze(2).expand(b, t, n, feats.shape[-1])  # (B, T, N, F)
    return torch.cat([value_channel, feats], dim=-1)


def future_time_features(marks: torch.Tensor, n: int) -> torch.Tensor:
    """Build a ``(B, T, N, F)`` tensor of future covariate features.

    Used by air-quality models that consume future covariates on the decoder
    side. The source depends on the rank of ``marks``:

    * ``(B, T, N, F)`` — node-structured future covariates from an
      air-quality dataset; used as-is.
    * ``(B, T, 6)`` — raw future calendar stamps; converted to
      ``[time_in_day, day_in_week]`` and broadcast across ``n`` nodes.

    Parameters
    ----------
    marks : torch.Tensor
        Future covariates ``(B, T, N, F)`` or raw stamps ``(B, T, 6)``.
    n : int
        Number of nodes to broadcast across (raw-stamp case only).

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(B, T, N, F)``.
    """
    if marks is not None and marks.dim() == 4:
        return marks
    feats = normalized_time_features(marks)  # (B, T, F)
    b, t, f = feats.shape
    return feats.unsqueeze(2).expand(b, t, n, f)


def coerce_time_length(marks: torch.Tensor, length: int) -> torch.Tensor:
    """Coerce a mark tensor to an exact temporal length.

    The air-quality models tie their future-covariate block to a fixed
    length (``seq_len`` / ``time_step``). The benchmark's decoder marks have
    length ``label_len + pred_len``, so we take the most recent ``length``
    future steps, repeating the last step if there are too few.

    Parameters
    ----------
    marks : torch.Tensor
        Marks of shape ``(B, L, ...)`` (raw stamps or node-structured
        covariates); only the time axis (dim 1) is adjusted.
    length : int
        Desired temporal length.

    Returns
    -------
    torch.Tensor
        Marks with the time axis coerced to ``length``.
    """
    have = marks.shape[1]
    if have == length:
        return marks
    if have > length:
        return marks[:, -length:]
    pad = marks[:, -1:].expand(
        marks.shape[0], length - have, *marks.shape[2:]
    )
    return torch.cat([marks, pad], dim=1)
