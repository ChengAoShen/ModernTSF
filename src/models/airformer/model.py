"""ModernTSF adapter for the AirFormer spatiotemporal forecasting model.

AirFormer uses causal temporal MSA with optional stochastic latent variables.
It consumes ``(B, T, N, F)`` and returns ``(B, horizon, N, output_dim)``
which is squeezed to ``(B, pred_len, N)``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models._external.marks import to_spatiotemporal
from models.airformer._upstream import AirFormer


class Model(nn.Module):
    """Adapter wrapping the upstream AirFormer model."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        cov_dim: int = 2,
        d_model: int = 32,
        nhead: int = 2,
        num_encoder_layers: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        input_dim = 1 + cov_dim
        self.pred_len = pred_len
        self.net = AirFormer(
            node_num=enc_in,
            input_dim=input_dim,
            output_dim=1,
            seq_len=seq_len,
            horizon=pred_len,
            dropout=dropout,
            spatial_flag=False,
            stochastic_flag=True,
            hidden_channels=d_model,
            end_channels=512,
            blocks=num_encoder_layers,
            mlp_expansion=2,
            num_heads=nhead,
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if x_mark_enc is None:
            x_mark_enc = x_enc.new_zeros((x_enc.shape[0], x_enc.shape[1], 6))
        st_input = to_spatiotemporal(x_enc, x_mark_enc)
        out = self.net(st_input)
        out = out.squeeze(-1)
        return out[:, :self.pred_len, :]
