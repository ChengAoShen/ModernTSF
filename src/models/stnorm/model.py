"""ModernTSF adapter for the STNorm spatiotemporal forecasting model.

STNorm (Wavenet-based) consumes (B, C, N, T) and outputs (B, out_dim, N, 1).
The adapter converts from the ModernTSF convention using to_spatiotemporal.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models._external.marks import TIME_FEATURES, to_spatiotemporal
from models.stnorm._upstream import STNorm


class Model(nn.Module):
    """Adapter wrapping the upstream STNorm model."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        cov_dim: int | None = None,
        channels: int = 16,
        kernel_size: int = 2,
        blocks: int = 8,
        layers: int = 2,
    ) -> None:
        super().__init__()
        cov = TIME_FEATURES if cov_dim is None else cov_dim
        in_dim = 1 + cov  # value + covariates
        self.net = STNorm(
            num_nodes=enc_in,
            node_num=enc_in,
            input_dim=in_dim,
            output_dim=1,
            seq_len=seq_len,
            horizon=pred_len,
            in_dim=in_dim,
            out_dim=pred_len,
            channels=channels,
            kernel_size=kernel_size,
            blocks=blocks,
            layers=layers,
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

        Returns (B, pred_len, N).
        """
        if x_mark_enc is None:
            x_mark_enc = x_enc.new_zeros(
                (x_enc.shape[0], x_enc.shape[1], 6))
        st = to_spatiotemporal(x_enc, x_mark_enc)  # (B, T, N, 1+F)
        # STNorm expects (B, C, N, T)
        x = st.permute(0, 3, 2, 1)  # (B, 1+F, N, T)
        out = self.net(x)  # (B, pred_len, N, 1)
        return out.squeeze(-1)  # (B, pred_len, N)
