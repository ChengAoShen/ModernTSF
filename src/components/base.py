"""Shared base class for ported external spatiotemporal / air-quality models.

The upstream repositories all inherit from a tiny ``BaseModel`` that simply
stores ``node_num / input_dim / output_dim / seq_len / horizon`` on ``self``.
It is reproduced here verbatim so the vendored model code can be dropped in
with only its import path changed.
"""

from __future__ import annotations

import torch.nn as nn


class BaseModel(nn.Module):
    """Minimal base model storing common forecasting dimensions."""

    def __init__(self, node_num, input_dim, output_dim, seq_len=12, horizon=12):
        super().__init__()
        self.node_num = node_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.horizon = horizon

    def param_num(self):
        """Return the total number of parameters."""
        return sum(param.nelement() for param in self.parameters())
