"""ModernTSF adapter for the STGCN spatiotemporal forecasting model.

STGCN uses Chebyshev graph convolutions with temporal convolutions.
It consumes ``(B, T, N, F)`` and returns ``(B, 1, N, output_dim)``
which is reshaped to ``(B, horizon, N)``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models._external.graph_utils import normalize_adj_mx
from models._external.marks import to_spatiotemporal
from models.stgcn._upstream import STGCN


class Model(nn.Module):
    """Adapter wrapping the upstream STGCN model."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        adj_mx: np.ndarray | None = None,
        cov_dim: int = 2,
        Ks: int = 3,
        Kt: int = 3,
        blocks: list | None = None,
        drop_prob: float = 0.0,
    ) -> None:
        super().__init__()
        if adj_mx is None:
            adj_mx = np.ones((enc_in, enc_in), dtype=np.float32)
        # Compute scaled Laplacian for Chebyshev graph conv
        L_list = normalize_adj_mx(adj_mx, "scalap")
        L = L_list[0]
        gso = torch.tensor(L, dtype=torch.float32)

        input_dim = 1 + cov_dim
        if blocks is None:
            blocks = [
                [input_dim],
                [input_dim, 32, 64],
                [64, 32, 128],
                [128, 128],
                [pred_len],
            ]
        self.net = STGCN(
            gso=gso,
            blocks=blocks,
            Kt=Kt,
            Ks=Ks,
            dropout=drop_prob,
            node_num=enc_in,
            input_dim=input_dim,
            output_dim=1,
            seq_len=seq_len,
            horizon=pred_len,
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
        # STGCN output: (B, pred_len, N, 1)
        out = self.net(st_input)
        # Squeeze trailing dim
        if out.dim() == 4:
            out = out.squeeze(-1)
        return out[:, :self.pred_len, :]
