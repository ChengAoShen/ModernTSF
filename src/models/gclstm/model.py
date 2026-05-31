"""ModernTSF adapter for the GCLSTM spatiotemporal forecasting model.

GCLSTM uses Chebyshev graph convolutions with LSTM cells.
It consumes ``(B, T, N, F)`` and returns ``(B, horizon, N, output_dim)``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models._external.graph_utils import normalize_adj_mx
from models._external.marks import to_spatiotemporal
from models.gclstm._upstream import GCLSTM


class Model(nn.Module):
    """Adapter wrapping the upstream GCLSTM model."""

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
        # Compute scaled Laplacian for Chebyshev conv
        L_list = normalize_adj_mx(adj_mx, "scalap")
        L = L_list[0]
        gso = torch.tensor(L, dtype=torch.float32)

        input_dim = 1 + cov_dim
        self.net = GCLSTM(
            gso=gso,
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
        if x_mark_enc is None:
            x_mark_enc = x_enc.new_zeros(
                (x_enc.shape[0], x_enc.shape[1], 6))
        st_input = to_spatiotemporal(x_enc, x_mark_enc)
        # GCLSTM output: (B, horizon, N, output_dim)
        out = self.net(st_input)
        if out.dim() == 4:
            out = out.squeeze(-1)
        return out[:, :self.pred_len, :]
