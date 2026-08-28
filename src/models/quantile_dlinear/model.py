"""Quantile DLinear: wraps the point DLinear backbone with a monotone quantile head."""

from __future__ import annotations

import torch.nn as nn

from components.dlinear import DLinearBackbone
from components.quantile_head import QuantileHead, validate_quantile_levels


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
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features
        self.c_out = 1 if features == "MS" else enc_in
        self.output_type = "quantile"
        levels = validate_quantile_levels(quantile_levels)
        self.backbone = DLinearBackbone(
            c_in=enc_in,
            seq_len=seq_len,
            pred_len=pred_len,
            kernel_size=kernel_size,
            individual=individual,
        )
        self.quantile_head = QuantileHead(levels, in_features=1)

    def forward(self, x, *args):
        if x.ndim != 3 or x.shape[1] != self.seq_len:
            raise ValueError(f"expected [batch, {self.seq_len}, channels], got {tuple(x.shape)}")
        base = self.backbone(x)               # (B, pred_len, enc_in)
        if self.features == "MS":
            base = base[:, :, -1:]
        base = base[:, -self.pred_len:, :]     # (B, pred_len, c_out)
        return self.quantile_head(base.unsqueeze(-1))  # (B, pred_len, c_out, Q)
