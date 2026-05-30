"""FreTS model implementation.

Vendored/adapted from https://github.com/thuml/Time-Series-Library
(models/FreTS.py), MIT License.

FreTS: Frequency-domain MLPs are More Effective Learners in Time Series
Forecasting (NeurIPS 2023).

Adapted for ModernTSF: the upstream ``configs``-object constructor is replaced
with plain keyword arguments, the string ``channel_independence`` flag is mapped
to a boolean kwarg, and only the long-term forecast path is kept. FreTS is
self-contained (no shared composite blocks), using only ``torch.fft`` ops.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        features="M",
        embed_size=128,
        hidden_size=256,
        channel_independence=False,
    ):
        super().__init__()
        self.features = features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_size = enc_in  # channels
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.channel_independence = channel_independence
        self.sparsity_threshold = 0.01
        self.scale = 0.02

        self.embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.r1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i1 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib1 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.r2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.i2 = nn.Parameter(self.scale * torch.randn(self.embed_size, self.embed_size))
        self.rb2 = nn.Parameter(self.scale * torch.randn(self.embed_size))
        self.ib2 = nn.Parameter(self.scale * torch.randn(self.embed_size))

        self.fc = nn.Sequential(
            nn.Linear(self.seq_len * self.embed_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.pred_len),
        )

    # dimension extension
    def tokenEmb(self, x):
        # x: [Batch, Input length, Channel]
        x = x.permute(0, 2, 1)
        x = x.unsqueeze(3)
        # N*T*1 x 1*D = N*T*D
        y = self.embeddings
        return x * y

    # frequency temporal learner
    def MLP_temporal(self, x, B, N, L):
        # [B, N, T, D]
        x = torch.fft.rfft(x, dim=2, norm="ortho")  # FFT on L dimension
        y = self.FreMLP(B, N, L, x, self.r2, self.i2, self.rb2, self.ib2)
        x = torch.fft.irfft(y, n=self.seq_len, dim=2, norm="ortho")
        return x

    # frequency channel learner
    def MLP_channel(self, x, B, N, L):
        # [B, N, T, D]
        x = x.permute(0, 2, 1, 3)
        # [B, T, N, D]
        x = torch.fft.rfft(x, dim=2, norm="ortho")  # FFT on N dimension
        y = self.FreMLP(B, L, N, x, self.r1, self.i1, self.rb1, self.ib1)
        x = torch.fft.irfft(y, n=self.feature_size, dim=2, norm="ortho")
        x = x.permute(0, 2, 1, 3)
        # [B, N, T, D]
        return x

    # frequency-domain MLPs
    def FreMLP(self, B, nd, dimension, x, r, i, rb, ib):
        o1_real = F.relu(
            torch.einsum("bijd,dd->bijd", x.real, r)
            - torch.einsum("bijd,dd->bijd", x.imag, i)
            + rb
        )

        o1_imag = F.relu(
            torch.einsum("bijd,dd->bijd", x.imag, r)
            + torch.einsum("bijd,dd->bijd", x.real, i)
            + ib
        )

        y = torch.stack([o1_real, o1_imag], dim=-1)
        y = F.softshrink(y, lambd=self.sparsity_threshold)
        y = torch.view_as_complex(y)
        return y

    def forecast(self, x_enc):
        # x: [Batch, Input length, Channel]
        B, T, N = x_enc.shape
        # embedding x: [B, N, T, D]
        x = self.tokenEmb(x_enc)
        bias = x
        # [B, N, T, D]
        if not self.channel_independence:
            x = self.MLP_channel(x, B, N, T)
        # [B, N, T, D]
        x = self.MLP_temporal(x, B, N, T)
        x = x + bias
        x = self.fc(x.reshape(B, N, -1)).permute(0, 2, 1)
        return x

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len :, :]  # [B, L, D]
