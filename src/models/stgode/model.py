"""ModernTSF adapter for the STGODE spatiotemporal forecasting model.

STGODE uses graph ODE with temporal convolutions.
It consumes ``(B, T, N, F)`` and returns ``(B, horizon, N, 1)``
which is squeezed to ``(B, pred_len, N)``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models._external.graph_utils import normalize_adj_mx
from models._external.marks import to_spatiotemporal
from models.stgode._upstream import STGODE


class Model(nn.Module):
    """Adapter wrapping the upstream STGODE model."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        adj_mx: np.ndarray | None = None,
        cov_dim: int = 2,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        if adj_mx is None:
            adj_mx = np.eye(enc_in, dtype=np.float32)
        # Compute doubletransition for spatial adj
        sp_list = normalize_adj_mx(adj_mx, "doubletransition")
        A_sp = torch.tensor(sp_list[0], dtype=torch.float32)
        # Use identity for semantic adj (no semantic info available)
        A_se = torch.tensor(
            np.eye(enc_in, dtype=np.float32), dtype=torch.float32)

        input_dim = 1 + cov_dim
        self.pred_len = pred_len
        self.net = STGODE(
            A_sp=A_sp,
            A_se=A_se,
            node_num=enc_in,
            input_dim=input_dim,
            output_dim=1,
            seq_len=seq_len,
            horizon=pred_len,
            num_layers=num_layers,
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

        Returns
        -------
        torch.Tensor
            Forecast of shape ``(B, pred_len, N)``.
        """
        if x_mark_enc is None:
            x_mark_enc = x_enc.new_zeros(
                (x_enc.shape[0], x_enc.shape[1], 6))
        st_input = to_spatiotemporal(x_enc, x_mark_enc)
        # out: (B, horizon, N, 1)
        out = self.net(st_input)
        # squeeze trailing dim
        out = out.squeeze(-1)  # (B, horizon, N)
        return out[:, :self.pred_len, :]
