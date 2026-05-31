"""ModernTSF adapter for the MoFo time-series forecasting model.

MoFo (https://github.com/PoorOtterBob/MoFo) is a univariate-per-channel
periodic transformer. Its ``forward(x_enc, x_mark_enc, x_dec, x_mark_dec)``
signature already matches ModernTSF's calling convention, but it reads the
calendar marks in the TFB normalization (features centered on ``[-0.5, 0.5]``)
and only supports ``periodic`` in ``{24, 96, 144, 288}``.

ModernTSF instead provides raw integer marks
``[year, month, day, weekday, hour, minute]``. This adapter rebuilds the
small slice of ``x_mark_enc`` that MoFo actually consumes so the periodic
position is recovered correctly regardless of the upstream normalization.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models._external.marks import _HOUR, _MINUTE, _WEEKDAY
from models.mofo._upstream import MoFo


class _MoFoConfig:
    """Lightweight config object matching MoFo's ``configs`` attribute access."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int,
        periodic: int,
        head: int,
        d_layers: int,
        bias: int,
        cias: int,
    ) -> None:
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.d_model = d_model
        self.periodic = periodic
        self.head = head
        self.d_layers = d_layers
        self.bias = bias
        self.cias = cias


class Model(nn.Module):
    """Adapter wrapping the upstream MoFo model.

    Parameters
    ----------
    seq_len : int
        Input sequence length.
    pred_len : int
        Forecast horizon.
    enc_in : int
        Number of input channels (variables).
    d_model : int
        Hidden dimension.
    periodic : int
        Period length (samples per cycle). Supported: 24, 96, 144, 288.
    head : int
        Number of attention heads (must divide ``d_model``).
    d_layers : int
        Number of MoFo backbone layers.
    bias : int
        Whether to use the channel-shared bias term.
    cias : int
        Whether to use the cyclic index-aware shift term.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        periodic: int = 24,
        head: int = 4,
        d_layers: int = 1,
        bias: int = 1,
        cias: int = 1,
    ) -> None:
        super().__init__()
        self.periodic = periodic
        config = _MoFoConfig(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            d_model=d_model,
            periodic=periodic,
            head=head,
            d_layers=d_layers,
            bias=bias,
            cias=cias,
        )
        self.net = MoFo(config)

    def _build_marks(self, marks: torch.Tensor) -> torch.Tensor:
        """Rebuild the TFB-normalized marks MoFo expects from raw marks.

        MoFo derives its periodic position from ``x_mark_enc`` using the TFB
        convention ``round((feature + 0.5) * (range - 1))``. We populate the
        few columns it reads so the recovered hour / minute / weekday match
        the true calendar values.

        Parameters
        ----------
        marks : torch.Tensor
            Raw marks of shape ``(B, T, 6)``.

        Returns
        -------
        torch.Tensor
            Synthetic marks of shape ``(B, T, 6)`` in TFB normalization.
        """
        hour = marks[..., _HOUR]
        minute = marks[..., _MINUTE]
        weekday = marks[..., _WEEKDAY]

        synth = torch.zeros_like(marks)
        # Column 0: hour-of-day (used when periodic == 24).
        synth[..., 0] = hour / 23.0 - 0.5
        # Column 1: minute-of-hour (used when periodic in {96, 144, 288}).
        synth[..., 1] = minute / 59.0 - 0.5
        # Column 2: hour-of-day for the multi-period branches.
        synth[..., 2] = hour / 23.0 - 0.5
        # Column 3: day-of-week (used for periodic_positionW).
        synth[..., 3] = weekday
        return synth

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
            Decoder values (unused by MoFo's forecast path).
        x_mark_dec : torch.Tensor, optional
            Raw decoder marks (unused).
        mask : torch.Tensor, optional
            Unused.

        Returns
        -------
        torch.Tensor
            Forecast of shape ``(B, pred_len, N)``.
        """
        if x_mark_enc is None:
            marks = x_enc.new_zeros((x_enc.shape[0], x_enc.shape[1], 6))
        else:
            marks = self._build_marks(x_mark_enc)
        return self.net(x_enc, marks, x_dec, x_mark_dec, mask)
