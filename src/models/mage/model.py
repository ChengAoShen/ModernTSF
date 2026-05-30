"""ModernTSF adapter for the MAGE spatiotemporal forecasting model.

MAGE (https://github.com/PoorOtterBob/MAGE) is a Mixture-of-Adaptive-Graph-
Experts model. It consumes ``(B, T, N, input_dim)`` where channel 0 is the
value and the remaining channels are normalized calendar features. Its
``Prompt_Pool`` reads the enabled temporal fields in a fixed order, each
indexing channel ``i + feature_dim``. With ``input_dim = 3`` we get
``feature_dim = 1``, so enabling exactly the ``minute`` (time-of-day) and
``weekday`` (day-of-week) embeddings makes the pool read channel 1
(``time_in_day``) and channel 2 (``day_in_week``) respectively.

The upstream model hard-codes a fixed batch size (``self.args.bs``) inside its
channel (de)compression reshapes; since ModernTSF's loader does not drop the
last (smaller) batch, this adapter refreshes that value on every forward.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from models._external.marks import TIME_FEATURES, to_calendar_spatiotemporal
from models.mage._upstream import MAGE


class Model(nn.Module):
    """Adapter wrapping the upstream MAGE model.

    Parameters
    ----------
    seq_len : int
        Input sequence length.
    pred_len : int
        Forecast horizon.
    enc_in : int
        Number of spatial nodes (channels).
    model_dim : int
        Expert hidden dimension.
    recur_num : int
        Number of graph experts.
    blocknum : int
        Number of stacked MAGE blocks (fixed at 3 upstream).
    topk : int
        Number of experts routed per node.
    node_dim : int
        Adaptive-graph generation dimension.
    tod_size : int
        Number of samples per day (time-of-day vocabulary size).
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        model_dim: int = 64,
        recur_num: int = 8,
        blocknum: int = 3,
        topk: int = 2,
        node_dim: int = 16,
        tod_size: int = 24,
    ) -> None:
        super().__init__()
        input_dim = 1 + TIME_FEATURES  # value + [time_in_day, day_in_week]
        model_args = SimpleNamespace(
            model_dim=model_dim,
            recur_num=recur_num,
            blocknum=blocknum,
            topk=topk,
            node_dim=node_dim,
            node_num=enc_in,
            # feature_dim = input_dim - 2 (the upstream main.py convention).
            feature_dim=input_dim - 2,
            bs=1,  # refreshed dynamically per forward.
            # Temporal embedding pool: enable exactly time-of-day + day-of-week.
            second=0,
            minute=tod_size,
            hour=0,
            day=0,
            week=0,
            weekday=7,
            month=0,
            quarter=0,
            year=0,
        )
        self.net = MAGE(
            model_args,
            node_num=enc_in,
            input_dim=input_dim,
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
        x_dec, x_mark_dec, mask
            Unused by MAGE.

        Returns
        -------
        torch.Tensor
            Forecast of shape ``(B, pred_len, N)``.
        """
        if x_mark_enc is None:
            x_mark_enc = x_enc.new_zeros((x_enc.shape[0], x_enc.shape[1], 6))
        st_input = to_calendar_spatiotemporal(x_enc, x_mark_enc)  # (B, T, N, 3)
        # The upstream reshapes use a fixed batch size; refresh it each call.
        self.net.args.bs = st_input.shape[0]
        out = self.net(st_input)
        if isinstance(out, tuple):  # training mode returns (pred, topk_indices)
            out = out[0]
        return out.squeeze(-1)  # (B, horizon, N, 1) -> (B, horizon, N)
