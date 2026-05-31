"""LSTM spatiotemporal baseline — verbatim from CauAir."""

import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    """Per-node LSTM with 1x1 conv embedding.

    Input: ``(B, T, N, F)`` — all F channels used.
    Output: ``(B, horizon, N, 1)``.
    """

    def __init__(
        self,
        input_dim: int,
        node_num: int,
        seq_len: int,
        horizon: int,
        init_dim: int = 32,
        hid_dim: int = 64,
        end_dim: int = 128,
        layer: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.horizon = horizon
        self.start_conv = nn.Conv2d(
            in_channels=input_dim, out_channels=init_dim, kernel_size=(1, 1)
        )
        self.lstm = nn.LSTM(
            input_size=init_dim,
            hidden_size=hid_dim,
            num_layers=layer,
            batch_first=True,
            dropout=dropout,
        )
        self.end_linear1 = nn.Linear(hid_dim, end_dim)
        self.end_linear2 = nn.Linear(end_dim, horizon)

    def forward(self, input, label=None):  # (b, t, n, f)
        x = input.transpose(1, 3)  # (b, f, n, t)
        b, f, n, t = x.shape
        x = x.transpose(1, 2).reshape(b * n, f, 1, t)
        x = self.start_conv(x).squeeze(2).transpose(1, 2)  # (b*n, t, init_dim)
        out, _ = self.lstm(x)
        x = out[:, -1, :]  # (b*n, hid_dim)
        x = F.relu(self.end_linear1(x))
        x = self.end_linear2(x)  # (b*n, horizon)
        x = x.reshape(b, n, self.horizon, 1).transpose(1, 2)  # (b, horizon, n, 1)
        return x
