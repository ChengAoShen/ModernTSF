"""ModernTSF adapter for the DCRNN spatiotemporal forecasting model.

DCRNN uses diffusion convolution with GRU-based encoder-decoder.
It consumes ``(B, T, N, F)`` and returns ``(B, horizon, N, output_dim)``
which is squeezed to ``(B, pred_len, N)``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models._external.marks import to_spatiotemporal
from models.dcrnn._upstream import DCRNN


class Model(nn.Module):
    """Adapter wrapping the upstream DCRNN model."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        adj_mx: np.ndarray | None = None,
        cov_dim: int = 2,
        n_filters: int = 64,
        max_diffusion_step: int = 2,
        filter_type: str = "doubletransition",
        num_rnn_layers: int = 2,
    ) -> None:
        super().__init__()
        if adj_mx is None:
            adj_mx = np.eye(enc_in, dtype=np.float32)
        input_dim = 1 + cov_dim
        self.pred_len = pred_len
        # Use a placeholder device; will be moved by .to()
        self.net = DCRNN(
            device=torch.device("cpu"),
            adj_mx=adj_mx,
            node_num=enc_in,
            input_dim=input_dim,
            output_dim=1,
            seq_len=seq_len,
            horizon=pred_len,
            n_filters=n_filters,
            max_diffusion_step=max_diffusion_step,
            filter_type=filter_type,
            num_rnn_layers=num_rnn_layers,
            use_curriculum_learning=False,
        )

    def _update_device(self) -> torch.device:
        """Detect current device and update internal references."""
        device = next(self.parameters()).device
        self.net.device = device
        # Rebuild supports on the correct device
        if self.net.supports and self.net.supports[0].device != device:
            self.net.supports = [s.to(device) for s in self.net.supports]
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
