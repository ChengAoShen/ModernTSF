"""ModernTSF adapter for the AirPhyNet spatiotemporal forecasting model.

AirPhyNet uses physics-informed Neural ODEs with graph convolution.
It consumes ``(B, T, N, F)`` and returns ``(B, horizon, N, output_dim)``
which is squeezed to ``(B, pred_len, N)``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models._external.marks import to_spatiotemporal
from models.airphynet._upstream import AirPhyNet


class Model(nn.Module):
    """Adapter wrapping the upstream AirPhyNet model."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        adj_mx: np.ndarray | None = None,
        cov_dim: int = 2,
        latent_dim: int = 4,
        rnn_units: int = 64,
        ode_method: str = "dopri5",
    ) -> None:
        super().__init__()
        if adj_mx is None:
            adj_mx = np.eye(enc_in, dtype=np.float32)
        input_dim = 1 + cov_dim
        self.pred_len = pred_len
        self.net = AirPhyNet(
            adj_mx=adj_mx,
            node_num=enc_in,
            input_dim=input_dim,
            output_dim=1,
            seq_len=seq_len,
            horizon=pred_len,
            latent_dim=latent_dim,
            rnn_units=rnn_units,
            ode_method=ode_method,
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
        # Move device for ODE
        device = x_enc.device
        self.net.device = device
        out = self.net(st_input)
        out = out.squeeze(-1)
        return out[:, :self.pred_len, :]
