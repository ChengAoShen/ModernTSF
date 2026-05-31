"""ModernTSF adapter for the SOFTS forecasting model.

SOFTS is a channel-independent model that takes (B, T, N) input and
outputs (B, pred_len, N). It treats each node as a separate variate.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.softs._upstream import SOFTS


class Model(nn.Module):
    """Adapter wrapping the upstream SOFTS model."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 128,
        d_ff: int = 256,
        e_layers: int = 2,
        dropout: float = 0.1,
        n_heads: int = 4,
        patch_len: int = 24,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        self.net = SOFTS(
            node_num=enc_in,
            input_dim=1,
            output_dim=1,
            seq_len=seq_len,
            horizon=pred_len,
            d_model=d_model,
            d_ff=d_ff,
            e_layers=e_layers,
            dropout=dropout,
            activation=activation,
            use_norm=True,
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

        Returns (B, pred_len, N).
        """
        # SOFTS takes (B, T, N) directly
        return self.net(x_enc, x_mark_enc)
