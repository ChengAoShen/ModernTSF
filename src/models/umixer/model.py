"""U-Mixer model implementation.

Vendored/adapted from https://github.com/XiangMa-Shaun/U-Mixer
(models/UMixer.py). The upstream repository ships no LICENSE file
(license: unspecified / no license file in source repo).

U-Mixer: An Unet-Mixer Architecture with Stationarity Correction for Time
Series Forecasting (AAAI 2024).

Adapted for ModernTSF:
- the upstream ``configs``-object constructor is replaced with plain keyword
  arguments;
- shared layers under ``models.module.*`` are reused (``PatchEmbedding`` from
  ``embed`` and ``RevIN`` from ``revin``);
- upstream hard-coded ``device='cuda:0'`` allocations are replaced with the
  input tensor's device so the model runs on CPU/GPU transparently;
- only the long-term forecasting path is kept.

The channel- / temporal-mixing MLP blocks and the stationarity-correction
helper are U-Mixer-specific and are kept local to this file.
"""

from __future__ import annotations

import torch
import torch.fft
import torch.nn as nn

from models.module.embed import PatchEmbedding
from models.module.revin import RevIN


def s_correction(x, x_pre):
    """Stationarity correction factor (alpha) between original and mixed."""
    x_fft = torch.fft.rfft(x, dim=1, norm="ortho")
    x_pre_fft = torch.fft.rfft(x_pre, dim=1, norm="ortho")
    x_fft = x_fft * torch.conj(x_fft)
    x_pre_fft = x_pre_fft * torch.conj(x_pre_fft)
    x_ifft = torch.fft.irfft(x_fft, dim=1)
    x_pre_ifft = torch.fft.irfft(x_pre_fft, dim=1)
    x_ifft = torch.clamp(x_ifft, min=0)
    x_pre_ifft = torch.clamp(x_pre_ifft, min=0)
    alpha = torch.sum(x_ifft * x_pre_ifft, dim=1, keepdim=True) / (
        torch.sum(x_pre_ifft * x_pre_ifft, dim=1, keepdim=True) + 0.001
    )
    return torch.sqrt(alpha)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class ChannelMix(nn.Module):
    """Channel-independent channel mixing over the d_model axis."""

    def __init__(self, d_model, patnum, dropout):
        super().__init__()
        self.conv1 = nn.ModuleList(nn.Linear(patnum, patnum) for _ in range(d_model))
        self.conv2 = nn.ModuleList(nn.Linear(patnum, patnum) for _ in range(d_model))
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.channels = d_model

    def forward(self, x):
        o = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
        for i in range(self.channels):
            o[:, :, i] = self.drop(self.conv2[i](self.gelu(self.conv1[i](x[:, :, i]))))
        res = o + x
        res = self.norm(res)
        return res


class TemporalMix(nn.Module):
    """Channel-independent temporal mixing over the patch axis."""

    def __init__(self, d_model, patnum, dropout):
        super().__init__()
        self.conv1 = nn.ModuleList(nn.Linear(d_model, d_model) for _ in range(patnum))
        self.conv2 = nn.ModuleList(nn.Linear(d_model, d_model) for _ in range(patnum))
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.channels = patnum

    def forward(self, x):
        o = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
        for i in range(self.channels):
            o[:, i, :] = self.drop(self.conv2[i](self.gelu(self.conv1[i](x[:, i, :]))))
        res = o + x
        res = self.norm(res)
        return res


class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        label_len=0,
        c_out=None,
        features="M",
        d_model=64,
        e_layers=2,
        patch_len=16,
        stride=8,
        dropout=0.1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features
        self.enc_in = enc_in
        self.c_out = c_out if c_out is not None else enc_in
        self.layer = e_layers

        self.layer_norm = nn.LayerNorm(d_model)
        self.predict_linear = nn.Linear(seq_len, pred_len + seq_len)

        self.Pnum = int((pred_len + seq_len - patch_len) / stride + 2)
        self.mlp_tempmix_md = nn.ModuleList(
            [TemporalMix(d_model, self.Pnum, dropout) for _ in range(e_layers)]
        )
        self.mlp_chanmix_md = nn.ModuleList(
            [ChannelMix(d_model, self.Pnum, dropout) for _ in range(e_layers)]
        )
        self.mlp_tempmix_mu = nn.ModuleList(
            [TemporalMix(d_model, self.Pnum, dropout) for _ in range(e_layers)]
        )
        self.mlp_chanmix_mu = nn.ModuleList(
            [ChannelMix(d_model, self.Pnum, dropout) for _ in range(e_layers)]
        )

        self.revin = RevIN(enc_in)
        # local PatchEmbedding signature: (d_model, patch_len, stride, padding, dropout)
        self.patch_embedding = PatchEmbedding(
            d_model, patch_len, stride, stride, dropout
        )
        self.head = FlattenHead(
            enc_in, d_model * self.Pnum, pred_len, head_dropout=dropout
        )
        self.comb = nn.Linear(e_layers, 1)

    def forecast(self, x_input, x_mark_input):
        x_ori = x_input.contiguous()
        x_input = self.revin(x_input, "norm")
        x_input = self.predict_linear(x_input.permute(0, 2, 1))
        x_input, n_vars = self.patch_embedding(x_input)

        x_old, _ = self.patch_embedding(x_ori.permute(0, 2, 1))

        x_all = torch.zeros(
            [x_input.shape[0], x_input.shape[1], x_input.shape[2], self.layer],
            device=x_input.device,
            dtype=x_input.dtype,
        )
        for i in range(self.layer):
            x_ud = self.mlp_tempmix_md[i](x_input)
            x_ud = self.mlp_chanmix_md[i](x_ud)
            for j in range(i, -1, -1):
                x_ud = self.mlp_tempmix_mu[j](x_ud)
                x_ud = self.mlp_chanmix_mu[j](x_ud)
            x_all[:, :, :, i] = x_ud
        x_input = self.comb(x_all).squeeze(-1)
        x_input = (
            s_correction(
                self.layer_norm(x_old),
                self.layer_norm(x_input[:, : x_old.shape[1], :]),
            )
            * x_input
        )
        x_input = torch.reshape(
            x_input, (-1, n_vars, x_input.shape[-2], x_input.shape[-1])
        )
        x_input = x_input.permute(0, 1, 3, 2)

        x_input = self.head(x_input)
        x_input = x_input.permute(0, 2, 1)
        x_input = self.revin(x_input, "denorm")

        return x_input[:, -self.pred_len :, :]

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        out = self.forecast(x_enc, x_mark_enc)
        return out[:, -self.pred_len :, :]  # [B, pred_len, c_out]
