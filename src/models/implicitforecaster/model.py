"""ModernTSF adapter for ImplicitForecaster.

This lightweight implementation follows the ModernTSF forecasting interface and
captures the main inductive bias of the verified open-source NeurIPS 2025
time-series forecasting work without vendoring its full training harness.
Upstream reference: https://github.com/rakuyorain/Implicit-Forecaster
"""

from __future__ import annotations

import torch.nn as nn

from models._recent_tsf import RecentTSFModel


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        dropout: float = 0.1,
        period: int = 24,
        num_prompts: int = 4,
        use_revin: bool = True,
    ) -> None:
        super().__init__()
        self.model = RecentTSFModel(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            variant="ImplicitForecaster",
            style="implicit",
            d_model=d_model,
            dropout=dropout,
            period=period,
            num_prompts=num_prompts,
            use_revin=use_revin,
        )

    def forward(self, x, *args):
        return self.model(x)
