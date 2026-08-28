"""Paper-driven local implementation of the LTSF-Linear baseline."""

from __future__ import annotations

import torch
from torch import nn

from components.channel_wise_linear import ChannelWiseLinear


class Model(nn.Module):
    """Apply one history-to-horizon affine map independently to each channel."""

    def __init__(self, c_in: int, seq_len: int, pred_len: int, individual: bool = False):
        super().__init__()
        if min(c_in, seq_len, pred_len) < 1:
            raise ValueError("channels and sequence lengths must be positive")
        self.c_in = c_in
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.projection = ChannelWiseLinear(seq_len, pred_len, c_in, individual)

    def forward(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor | None = None,
                x_dec: torch.Tensor | None = None, x_mark_dec: torch.Tensor | None = None,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        del x_mark_enc, x_dec, x_mark_dec, mask
        if x_enc.shape[1:] != (self.seq_len, self.c_in):
            raise ValueError("x_enc does not match configured time/channel dimensions")
        return self.projection(x_enc.transpose(1, 2)).transpose(1, 2)
