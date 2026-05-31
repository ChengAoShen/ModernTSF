"""ModernTSF adapter for the DGCRN spatiotemporal forecasting model.

DGCRN uses dynamic graph convolution with GRU-based encoder-decoder.
It consumes ``(B, T, N, F)`` and returns ``(B, horizon, N, output_dim)``
which is squeezed to ``(B, pred_len, N)``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models._external.graph_utils import adj_to_supports
from models._external.marks import to_spatiotemporal
from models.dgcrn._upstream import DGCRN


class Model(nn.Module):
    """Adapter wrapping the upstream DGCRN model."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        adj_mx: np.ndarray | None = None,
        cov_dim: int = 2,
        gcn_depth: int = 2,
        dropout: float = 0.3,
        subgraph_size: int = 20,
        node_dim: int = 40,
        rnn_size: int = 64,
        adj_type: str = "doubletransition",
    ) -> None:
        super().__init__()
        if adj_mx is None:
            adj_mx = np.eye(enc_in, dtype=np.float32)
        # Build predefined_adj as list of tensors
        supports = adj_to_supports(adj_mx, adj_type)
        input_dim = 1 + cov_dim
        self.pred_len = pred_len
        self.net = DGCRN(
            device=torch.device("cpu"),
            node_num=enc_in,
            input_dim=input_dim,
            output_dim=1,
            seq_len=seq_len,
            horizon=pred_len,
            predefined_adj=supports,
            gcn_depth=gcn_depth,
            rnn_size=rnn_size,
            node_dim=node_dim,
            dropout=dropout,
        )

    def _update_device(self) -> torch.device:
        """Detect current device and update internal references."""
        device = next(self.parameters()).device
        self.net.device = device
        self.net.idx = self.net.idx.to(device)
        if self.net.predefined_adj and self.net.predefined_adj[0].device != device:
            self.net.predefined_adj = [a.to(device) for a in self.net.predefined_adj]
        return device

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
        self._update_device()
        if x_mark_enc is None:
            x_mark_enc = x_enc.new_zeros(
                (x_enc.shape[0], x_enc.shape[1], 6))
        st_input = to_spatiotemporal(x_enc, x_mark_enc)
        # out: (B, horizon, N, output_dim)
        out = self.net(st_input)
        # squeeze output_dim=1
        out = out.squeeze(-1)  # (B, horizon, N)
        return out[:, :self.pred_len, :]
