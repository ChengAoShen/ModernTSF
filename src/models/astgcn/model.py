"""ModernTSF adapter for the ASTGCN spatiotemporal forecasting model.

ASTGCN uses attention-based spatial-temporal graph convolutions.
It consumes ``(B, T, N, F)`` and returns ``(B, horizon, N, 1)``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models._external.graph_utils import cheb_poly, normalize_adj_mx
from models._external.marks import to_spatiotemporal
from models.astgcn._upstream import ASTGCN


class Model(nn.Module):
    """Adapter wrapping the upstream ASTGCN model."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        adj_mx: np.ndarray | None = None,
        cov_dim: int = 2,
        nb_block: int = 2,
        K: int = 3,
        nb_chev_filter: int = 64,
        nb_time_filter: int = 64,
    ) -> None:
        super().__init__()
        if adj_mx is None:
            adj_mx = np.ones((enc_in, enc_in), dtype=np.float32)
        # Compute scaled Laplacian and Chebyshev polynomials
        L_list = normalize_adj_mx(adj_mx, "scalap")
        L = L_list[0]
        Lk = cheb_poly(L, K)  # (K, N, N)
        # Convert to list of tensors for cheb_conv_withSAt
        cheb_polynomials = [
            torch.tensor(Lk[i], dtype=torch.float32)
            for i in range(K)]

        input_dim = 1 + cov_dim
        self.net = ASTGCN(
            cheb_poly=cheb_polynomials,
            order=K,
            nb_block=nb_block,
            nb_chev_filter=nb_chev_filter,
            nb_time_filter=nb_time_filter,
            time_stride=1,
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
        # ASTGCN output: (B, horizon, N, 1)
        out = self.net(st_input)
        if out.dim() == 4:
            out = out.squeeze(-1)
        return out[:, :self.pred_len, :]
