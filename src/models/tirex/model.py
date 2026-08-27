"""ModernTSF adapter for TiRex (quantile output).

This lightweight implementation follows the ModernTSF forecasting interface and
captures the main inductive bias of the verified open-source NeurIPS 2025
time-series forecasting work without vendoring its full training harness.
Upstream reference: https://github.com/NX-AI/tirex

Phase 2: the point backbone's ``(B, pred_len, C)`` output is wrapped by the
shared monotone ``QuantileHead`` to emit a non-crossing rank-4 quantile tensor
``(B, pred_len, C, Q)`` ordered to match ``evaluation.quantile_levels``.
"""

from __future__ import annotations

import torch.nn as nn

from adapters.recent_tsf import RecentTSFModel
from components.quantile_head import QuantileHead

_DEFAULT_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        features: str = "M",
        d_model: int = 64,
        dropout: float = 0.1,
        period: int = 24,
        num_prompts: int = 4,
        use_revin: bool = True,
        quantile_levels: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.features = features
        self.c_out = 1 if features == "MS" else enc_in
        self.output_type = "quantile"
        levels = list(quantile_levels) if quantile_levels else _DEFAULT_LEVELS
        self.model = RecentTSFModel(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            variant="TiRex",
            style="portfolio",
            d_model=d_model,
            dropout=dropout,
            period=period,
            num_prompts=num_prompts,
            use_revin=use_revin,
        )
        self.quantile_head = QuantileHead(levels, in_features=1)

    def forward(self, x, *args):
        base = self.model(x)                  # (B, pred_len, enc_in)
        if self.features == "MS":
            base = base[:, :, -1:]            # (B, pred_len, 1)
        base = base[:, -self.pred_len:, :]    # (B, pred_len, c_out)
        return self.quantile_head(base.unsqueeze(-1))  # (B, pred_len, c_out, Q)
