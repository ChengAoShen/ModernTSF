"""TimeKAN model implementation.

Vendored/adapted from https://github.com/huangst21/TimeKAN
(models/TimeKAN.py and layers/ChebyKANLayer.py), Apache License 2.0.

TimeKAN: KAN-based Frequency Decomposition Learning Architecture for
Long-term Time Series Forecasting (ICLR 2025).

Adapted for ModernTSF: the upstream ``configs``-object constructor is replaced
with plain keyword arguments, only the long-term forecast path is kept, and the
shared layers under ``models.module.*`` are reused (``series_decomp``,
``DataEmbedding_wo_pos``, ``Normalize``). The KAN primitive
(``ChebyKANLinear``) is small and specific to TimeKAN, so it is vendored
locally below. The frequency decomposition / mixing blocks (CFD, M-KAN,
Frequency Mixing) are TimeKAN-specific and kept local to this file.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.module.autoformer_encdec import series_decomp
from models.module.embed import DataEmbedding_wo_pos
from models.module.standard_norm import Normalize


class ChebyKANLinear(nn.Module):
    """KAN linear layer using Chebyshev polynomials (vendored from upstream)."""

    def __init__(self, input_dim, output_dim, degree):
        super().__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree

        self.cheby_coeffs = nn.Parameter(
            torch.empty(input_dim, output_dim, degree + 1)
        )
        nn.init.normal_(
            self.cheby_coeffs, mean=0.0, std=1 / (input_dim * (degree + 1))
        )
        self.register_buffer("arange", torch.arange(0, degree + 1, 1))

    def forward(self, x):
        b, c_in = x.shape
        # Normalize to [-1, 1] and apply Chebyshev basis.
        x = x.view((b, c_in, 1)).expand(-1, -1, self.degree + 1)
        x = torch.tanh(x)
        x = torch.tanh(x)
        x = torch.acos(x)
        x = x * self.arange
        x = x.cos()
        y = torch.einsum("bid,iod->bo", x, self.cheby_coeffs)
        y = y.view(-1, self.outdim)
        return y


class ChebyKANLayer(nn.Module):
    def __init__(self, in_features, out_features, order):
        super().__init__()
        self.fc1 = ChebyKANLinear(in_features, out_features, order)

    def forward(self, x):
        B, N, C = x.shape
        x = self.fc1(x.reshape(B * N, C))
        x = x.reshape(B, N, -1).contiguous()
        return x


class BasicConv(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        kernel_size,
        degree,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        act=False,
        bn=False,
        bias=False,
        dropout=0.0,
    ):
        super().__init__()
        self.out_channels = c_out
        self.conv = nn.Conv1d(
            c_in,
            c_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm1d(c_out) if bn else None
        self.act = nn.GELU() if act else None
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.bn is not None:
            x = self.bn(x)
        x = self.conv(x.transpose(-1, -2)).transpose(-1, -2)
        if self.act is not None:
            x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return x


class M_KAN(nn.Module):
    """Multi-order KAN representation learning block."""

    def __init__(self, d_model, seq_len, order):
        super().__init__()
        self.channel_mixer = nn.Sequential(ChebyKANLayer(d_model, d_model, order))
        self.conv = BasicConv(
            d_model, d_model, kernel_size=3, degree=order, groups=d_model
        )

    def forward(self, x):
        x1 = self.channel_mixer(x)
        x2 = self.conv(x)
        return x1 + x2


def _frequency_interpolation(x, seq_len, target_len):
    len_ratio = seq_len / target_len
    x_fft = torch.fft.rfft(x, dim=2)
    out_fft = torch.zeros(
        [x_fft.size(0), x_fft.size(1), target_len // 2 + 1],
        dtype=x_fft.dtype,
    ).to(x_fft.device)
    out_fft[:, :, : seq_len // 2 + 1] = x_fft
    out = torch.fft.irfft(out_fft, dim=2)
    out = out * len_ratio
    return out


class FrequencyDecomp(nn.Module):
    """Cascaded Frequency Decomposition (CFD)."""

    def __init__(self, seq_len, down_sampling_window, down_sampling_layers):
        super().__init__()
        self.seq_len = seq_len
        self.down_sampling_window = down_sampling_window
        self.down_sampling_layers = down_sampling_layers

    def forward(self, level_list):
        level_list_reverse = level_list.copy()
        level_list_reverse.reverse()
        out_low = level_list_reverse[0]
        out_high = level_list_reverse[1]
        out_level_list = [out_low]
        for i in range(len(level_list_reverse) - 1):
            out_high_res = _frequency_interpolation(
                out_low.transpose(1, 2),
                self.seq_len
                // (self.down_sampling_window ** (self.down_sampling_layers - i)),
                self.seq_len
                // (
                    self.down_sampling_window
                    ** (self.down_sampling_layers - i - 1)
                ),
            ).transpose(1, 2)
            out_high_left = out_high - out_high_res
            out_low = out_high
            if i + 2 <= len(level_list_reverse) - 1:
                out_high = level_list_reverse[i + 2]
            out_level_list.append(out_high_left)
        out_level_list.reverse()
        return out_level_list


class FrequencyMixing(nn.Module):
    """Frequency Mixing with M-KAN blocks."""

    def __init__(
        self,
        d_model,
        seq_len,
        down_sampling_window,
        down_sampling_layers,
        begin_order,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.down_sampling_window = down_sampling_window
        self.down_sampling_layers = down_sampling_layers
        self.front_block = M_KAN(
            d_model,
            seq_len // (down_sampling_window ** down_sampling_layers),
            order=begin_order,
        )
        self.front_blocks = nn.ModuleList(
            [
                M_KAN(
                    d_model,
                    seq_len
                    // (down_sampling_window ** (down_sampling_layers - i - 1)),
                    order=i + begin_order + 1,
                )
                for i in range(down_sampling_layers)
            ]
        )

    def forward(self, level_list):
        level_list_reverse = level_list.copy()
        level_list_reverse.reverse()
        out_low = level_list_reverse[0]
        out_high = level_list_reverse[1]
        out_low = self.front_block(out_low)
        out_level_list = [out_low]
        for i in range(len(level_list_reverse) - 1):
            out_high = self.front_blocks[i](out_high)
            out_high_res = _frequency_interpolation(
                out_low.transpose(1, 2),
                self.seq_len
                // (self.down_sampling_window ** (self.down_sampling_layers - i)),
                self.seq_len
                // (
                    self.down_sampling_window
                    ** (self.down_sampling_layers - i - 1)
                ),
            ).transpose(1, 2)
            out_high = out_high + out_high_res
            out_low = out_high
            if i + 2 <= len(level_list_reverse) - 1:
                out_high = level_list_reverse[i + 2]
            out_level_list.append(out_low)
        out_level_list.reverse()
        return out_level_list


class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        label_len=0,
        features="M",
        c_out=None,
        d_model=16,
        e_layers=1,
        down_sampling_window=2,
        down_sampling_layers=1,
        begin_order=0,
        moving_avg=25,
        dropout=0.1,
        embed="timeF",
        freq="h",
        use_norm=1,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.features = features
        self.enc_in = enc_in
        self.c_out = c_out if c_out is not None else enc_in
        self.down_sampling_window = down_sampling_window
        self.down_sampling_layers = down_sampling_layers
        self.layer = e_layers

        self.res_blocks = nn.ModuleList(
            [
                FrequencyDecomp(
                    seq_len, down_sampling_window, down_sampling_layers
                )
                for _ in range(e_layers)
            ]
        )
        self.add_blocks = nn.ModuleList(
            [
                FrequencyMixing(
                    d_model,
                    seq_len,
                    down_sampling_window,
                    down_sampling_layers,
                    begin_order,
                )
                for _ in range(e_layers)
            ]
        )

        self.preprocess = series_decomp(moving_avg)
        self.enc_embedding = DataEmbedding_wo_pos(
            1, d_model, embed, freq, dropout
        )
        self.normalize_layers = nn.ModuleList(
            [
                Normalize(
                    enc_in, affine=True, non_norm=True if use_norm == 0 else False
                )
                for _ in range(down_sampling_layers + 1)
            ]
        )
        self.projection_layer = nn.Linear(d_model, 1, bias=True)
        self.predict_layer = nn.Linear(seq_len, pred_len)

    def __multi_level_process_inputs(self, x_enc):
        down_pool = nn.AvgPool1d(self.down_sampling_window)
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)
        x_enc_ori = x_enc
        x_enc_sampling_list = [x_enc.permute(0, 2, 1)]
        for _ in range(self.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling
        return x_enc_sampling_list

    def forecast(self, x_enc):
        x_enc = self.__multi_level_process_inputs(x_enc)
        x_list = []
        B = T = N = 0
        for i, x in enumerate(x_enc):
            B, T, N = x.size()
            x = self.normalize_layers[i](x, "norm")
            x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
            x_list.append(x)

        enc_out_list = []
        for x in x_list:
            enc_out = self.enc_embedding(x, None)  # [B*N,T,d_model]
            enc_out_list.append(enc_out)

        for i in range(self.layer):
            enc_out_list = self.res_blocks[i](enc_out_list)
            enc_out_list = self.add_blocks[i](enc_out_list)

        dec_out = enc_out_list[0]
        dec_out = self.predict_layer(dec_out.permute(0, 2, 1)).permute(0, 2, 1)
        dec_out = (
            self.projection_layer(dec_out)
            .reshape(B, self.c_out, self.pred_len)
            .permute(0, 2, 1)
            .contiguous()
        )
        dec_out = self.normalize_layers[0](dec_out, "denorm")
        return dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len :, :]  # [B, pred_len, c_out]
