"""ModernTSF adapter for the GWNet spatiotemporal forecasting model.

GWNet (Graph WaveNet) uses dilated causal convolutions with adaptive graph
learning. It consumes ``(B, T, N, F)`` and returns
``(B, output_dim*horizon, N, 1)`` which is reshaped to ``(B, horizon, N)``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models._external.graph_utils import adj_to_supports
from models._external.marks import to_spatiotemporal
from models.gwnet._upstream import GWNET


class Model(nn.Module):
    """Adapter wrapping the upstream GWNet model."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        adj_mx: np.ndarray | None = None,
        cov_dim: int = 2,
        residual_channels: int = 32,
        dilation_channels: int = 32,
        skip_channels: int = 64,
        end_channels: int = 128,
        dropout: float = 0.3,
        blocks: int = 8,
        layers: int = 2,
        adp_adj: bool = True,
        adj_type: str = "doubletransition",
    ) -> None:
        super().__init__()
        if adj_mx is None:
            adj_mx = np.ones((enc_in, enc_in), dtype=np.float32)
        supports = adj_to_supports(adj_mx, adj_type)
        input_dim = 1 + cov_dim
        self.net = GWNET(
            supports=supports,
            adp_adj=adp_adj,
            dropout=dropout,
            residual_channels=residual_channels,
            dilation_channels=dilation_channels,
            skip_channels=skip_channels,
            end_channels=end_channels,
            node_num=enc_in,
            input_dim=input_dim,
            output_dim=1,
            seq_len=seq_len,
            horizon=pred_len,
            blocks=blocks,
            layers=layers,
        )
        self.pred_len = pred_len

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forecast future values.

        Returns
        -------
        torch.Tensor
            Forecast of shape ``(B, pred_len, N)``.
        """
        if x_mark_enc is None:
            x_mark_enc = x_enc.new_zeros(
                (x_enc.shape[0], x_enc.shape[1], 6))
        st_input = to_spatiotemporal(x_enc, x_mark_enc)
        # out: (B, output_dim*horizon, N, 1)
        out = self.net(st_input)
        # reshape to (B, horizon, N)
        out = out.squeeze(-1)  # (B, horizon, N)
        return out[:, :self.pred_len, :]
