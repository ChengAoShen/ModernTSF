"""Historical Last baseline — verbatim from CauAir."""

import torch.nn as nn


class HL(nn.Module):
    """Repeat the last time step as the forecast (naive baseline).

    Input: ``(B, T, N, F)`` — only channel 0 (value) is used.
    Output: ``(B, horizon, N, 1)``.
    """

    def __init__(self, horizon: int, **kwargs):
        super().__init__()
        self.horizon = horizon
        # Dummy parameter so the model has at least one param (optimizer needs it).
        self.fake = nn.Linear(1, 1)

    def forward(self, x, label=None):  # (b, t, n, f)
        out = x[:, [-1], :, 0:1].expand(-1, self.horizon, -1, -1)
        # Add zero contribution from fake param so backward works.
        return out + 0.0 * self.fake(out[..., :1]).mean()
