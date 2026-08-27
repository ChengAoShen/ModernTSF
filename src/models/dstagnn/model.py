"""ModernTSF adapter for the DSTAGNN spatiotemporal forecasting model.

DSTAGNN uses spatial-temporal attention, graph convolution, and multi-scale
gated temporal convolution. The adapter consumes values shaped ``(B, T, N)``
and returns ``(B, pred_len, N)``; covariate marks are intentionally unused.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models.dstagnn._upstream import DSTAGNN


class Model(nn.Module):
    """Adapter wrapping the upstream DSTAGNN model."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        adj_mx: np.ndarray | None = None,
        d_model: int = 64,
        d_k: int = 8,
        d_v: int = 8,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        if adj_mx is None:
            adj_mx = np.eye(enc_in, dtype=np.float32)
        # DSTAGNN works best with input_dim=1 (univariate per node)
        input_dim = 1
        self.pred_len = pred_len
        self.net = DSTAGNN(
            adj_mx=adj_mx,
            node_num=enc_in,
            input_dim=input_dim,
            output_dim=1,
            seq_len=seq_len,
            horizon=pred_len,
            d_model=d_model,
            d_k=d_k,
            d_v=d_v,
            n_heads=n_heads,
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # DSTAGNN uses only the value channel (input_dim=1)
        # x_enc: (B, T, N) -> (B, T, N, 1)
        st_input = x_enc.unsqueeze(-1)
        out = self.net(st_input)
        out = out.squeeze(-1)
        return out[:, :self.pred_len, :]
