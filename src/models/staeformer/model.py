"""ModernTSF adapter for the STAEformer spatiotemporal model.

STAEformer consumes (B, T, N, C) and outputs (B, out_steps, N, output_dim).
The adapter uses to_spatiotemporal to build the input tensor.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models._external.marks import TIME_FEATURES, to_spatiotemporal
from models.staeformer._upstream import STAEformer


class Model(nn.Module):
    """Adapter wrapping the upstream STAEformer model."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        cov_dim: int | None = None,
        input_embedding_dim: int = 24,
        tod_embedding_dim: int = 24,
        dow_embedding_dim: int = 24,
        adaptive_embedding_dim: int = 56,
        feed_forward_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        steps_per_day: int = 24,
    ) -> None:
        super().__init__()
        cov = TIME_FEATURES if cov_dim is None else cov_dim
        input_dim = 1 + cov
        self.net = STAEformer(
            node_num=enc_in,
            input_dim=input_dim,
            output_dim=1,
            seq_len=seq_len,
            horizon=pred_len,
            in_steps=seq_len,
            out_steps=pred_len,
            steps_per_day=steps_per_day,
            input_embedding_dim=input_embedding_dim,
            tod_embedding_dim=tod_embedding_dim,
            dow_embedding_dim=dow_embedding_dim,
            spatial_embedding_dim=0,
            adaptive_embedding_dim=adaptive_embedding_dim,
            feed_forward_dim=feed_forward_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
            use_mixed_proj=True,
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
        out = self.net(st)  # (B, out_steps, N, output_dim)
        return out.squeeze(-1)  # (B, pred_len, N)
