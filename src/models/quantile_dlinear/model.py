"""Quantile DLinear: wraps the point DLinear backbone with a monotone quantile head."""

from __future__ import annotations

import torch.nn as nn

from components.dlinear import DLinearBackbone
from components.quantile_head import QuantileHead

_DEFAULT_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        features: str = "M",
        kernel_size: int = 25,
        individual: bool = False,
        quantile_levels: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.features = features
        self.c_out = 1 if features == "MS" else enc_in
        self.output_type = "quantile"
        levels = list(quantile_levels) if quantile_levels else _DEFAULT_LEVELS
        self.backbone = DLinearBackbone(
            c_in=enc_in,
            seq_len=seq_len,
            pred_len=pred_len,
            kernel_size=kernel_size,
            individual=individual,
        )
        self.quantile_head = QuantileHead(levels, in_features=1)

    def forward(self, x, *args):
        base = self.backbone(x)               # (B, pred_len, enc_in)
        if self.features == "MS":
            base = base[:, :, -1:]
        base = base[:, -self.pred_len:, :]     # (B, pred_len, c_out)
        return self.quantile_head(base.unsqueeze(-1))  # (B, pred_len, c_out, Q)
