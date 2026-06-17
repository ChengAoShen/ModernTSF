"""GaussianMLP: a flatten-MLP Gaussian distribution forecaster.

Flattens the input window and maps it through an MLP to per-(horizon, channel)
Gaussian parameters (loc, scale > 0), emitting a rank-4 (B, pred_len, C, 2)
tensor consumed by the Phase 1 nll_gaussian loss and Gaussian CRPS metrics.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        features: str = "M",
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.features = features
        self.c_out = 1 if features == "MS" else enc_in
        self.eps = eps
        # Probabilistic output contract (Phase 1): emit (loc, scale) per element.
        self.output_type = "distribution"
        self.distribution_family = "gaussian"

        in_dim = seq_len * enc_in
        out_dim = pred_len * self.c_out
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(max(num_layers, 1)):
            layers += [nn.Linear(prev, hidden_size), nn.ReLU(), nn.Dropout(dropout)]
            prev = hidden_size
        self.backbone = nn.Sequential(*layers)
        self.loc_head = nn.Linear(prev, out_dim)
        self.scale_head = nn.Linear(prev, out_dim)

    def forward(self, x, *args):
        # x: (B, seq_len, enc_in)
        B = x.shape[0]
        h = self.backbone(x.reshape(B, -1))             # (B, hidden)
        loc = self.loc_head(h).reshape(B, self.pred_len, self.c_out)
        scale = F.softplus(self.scale_head(h)).reshape(B, self.pred_len, self.c_out)
        scale = scale + self.eps                         # strictly > 0
        return torch.stack([loc, scale], dim=-1)         # (B, pred_len, c_out, 2)
