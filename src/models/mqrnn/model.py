"""MQ-RNN: a multi-horizon quantile RNN seq2seq forecaster.

A shared GRU encodes each channel's history into a context vector; a global MLP
decoder expands the context into a per-(horizon, channel) base representation
that the monotone QuantileHead turns into a non-crossing (B, pred_len, C, Q)
grid. Channels are treated as independent series sharing one encoder/decoder
(the DeepAR per-node convention).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models._quantile_head import QuantileHead

_DEFAULT_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        features: str = "M",
        hidden_size: int = 64,
        num_layers: int = 1,
        decoder_hidden: int = 64,
        dropout: float = 0.1,
        quantile_levels: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.features = features
        self.c_out = 1 if features == "MS" else enc_in
        self.output_type = "quantile"
        levels = list(quantile_levels) if quantile_levels else _DEFAULT_LEVELS

        # Shared GRU encoder; each channel is fed as a length-seq_len scalar
        # series -> (B*C, seq_len, 1).
        self.encoder = nn.GRU(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # Global MLP decoder: context (hidden_size) -> per-horizon base scalar.
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, decoder_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_hidden, pred_len),
        )
        self.quantile_head = QuantileHead(levels, in_features=1)

    def forward(self, x, *args):
        # x: (B, seq_len, enc_in)
        B, _, C = x.shape
        # Per-channel independent series: (B, L, C) -> (B*C, L, 1)
        seq = x.permute(0, 2, 1).reshape(B * C, self.seq_len, 1)
        _, h = self.encoder(seq)               # h: (num_layers, B*C, hidden)
        ctx = h[-1]                            # (B*C, hidden)
        base = self.decoder(ctx)               # (B*C, pred_len)
        base = base.reshape(B, C, self.pred_len).permute(0, 2, 1)  # (B, pred_len, C)
        if self.features == "MS":
            base = base[:, :, -1:]             # (B, pred_len, 1)
        return self.quantile_head(base.unsqueeze(-1))  # (B, pred_len, c_out, Q)
