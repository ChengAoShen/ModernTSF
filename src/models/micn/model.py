"""MICN model implementation.

Vendored/adapted from https://github.com/thuml/Time-Series-Library
(models/MICN.py), MIT License.

MICN: Multi-scale Local and Global Context Modeling for Long-term Series
Forecasting (ICLR 2023).

Adapted for ModernTSF: the upstream ``configs``-object constructor is replaced
with plain keyword arguments, all non-forecasting task branches
(imputation/anomaly/classification) are dropped, and the shared decomposition
and embedding layers under ``models.module.*`` are reused
(``series_decomp``, ``series_decomp_multi``, ``DataEmbedding``). The
``MIC`` / ``SeasonalPrediction`` blocks (multi-scale isometric convolution)
are MICN-specific and kept local to this file. The upstream hard-coded CUDA
device is replaced with a device-agnostic implementation that follows the
input tensor's device.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.module.autoformer_encdec import series_decomp, series_decomp_multi
from models.module.embed import DataEmbedding


class MIC(nn.Module):
    """MIC layer to extract local and global features."""

    def __init__(
        self,
        feature_size=512,
        n_heads=8,
        dropout=0.05,
        decomp_kernel=(32,),
        conv_kernel=(24,),
        isometric_kernel=(18, 6),
    ):
        super().__init__()
        self.conv_kernel = list(conv_kernel)

        # isometric convolution
        self.isometric_conv = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=feature_size,
                    out_channels=feature_size,
                    kernel_size=i,
                    padding=0,
                    stride=1,
                )
                for i in isometric_kernel
            ]
        )

        # downsampling convolution: padding=i//2, stride=i
        self.conv = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=feature_size,
                    out_channels=feature_size,
                    kernel_size=i,
                    padding=i // 2,
                    stride=i,
                )
                for i in conv_kernel
            ]
        )

        # upsampling convolution
        self.conv_trans = nn.ModuleList(
            [
                nn.ConvTranspose1d(
                    in_channels=feature_size,
                    out_channels=feature_size,
                    kernel_size=i,
                    padding=0,
                    stride=i,
                )
                for i in conv_kernel
            ]
        )

        self.decomp = nn.ModuleList([series_decomp(k) for k in decomp_kernel])
        self.merge = nn.Conv2d(
            in_channels=feature_size,
            out_channels=feature_size,
            kernel_size=(len(self.conv_kernel), 1),
        )

        # feedforward network
        self.conv1 = nn.Conv1d(
            in_channels=feature_size, out_channels=feature_size * 4, kernel_size=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=feature_size * 4, out_channels=feature_size, kernel_size=1
        )
        self.norm1 = nn.LayerNorm(feature_size)
        self.norm2 = nn.LayerNorm(feature_size)

        self.norm = nn.LayerNorm(feature_size)
        self.act = nn.Tanh()
        self.drop = nn.Dropout(dropout)

    def conv_trans_conv(self, input, conv1d, conv1d_trans, isometric):
        _, seq_len, _ = input.shape
        x = input.permute(0, 2, 1)

        # downsampling convolution
        x1 = self.drop(self.act(conv1d(x)))
        x = x1

        # isometric convolution
        zeros = torch.zeros(
            (x.shape[0], x.shape[1], x.shape[2] - 1),
            device=x.device,
            dtype=x.dtype,
        )
        x = torch.cat((zeros, x), dim=-1)
        x = self.drop(self.act(isometric(x)))
        x = self.norm((x + x1).permute(0, 2, 1)).permute(0, 2, 1)

        # upsampling convolution
        x = self.drop(self.act(conv1d_trans(x)))
        x = x[:, :, :seq_len]  # truncate

        x = self.norm(x.permute(0, 2, 1) + input)
        return x

    def forward(self, src):
        # multi-scale
        multi = []
        for i in range(len(self.conv_kernel)):
            src_out, _ = self.decomp[i](src)
            src_out = self.conv_trans_conv(
                src_out, self.conv[i], self.conv_trans[i], self.isometric_conv[i]
            )
            multi.append(src_out)

        # merge
        mg = torch.cat([m.unsqueeze(1) for m in multi], dim=1)
        mg = self.merge(mg.permute(0, 3, 1, 2)).squeeze(-2).permute(0, 2, 1)

        y = self.norm1(mg)
        y = self.conv2(self.conv1(y.transpose(-1, 1))).transpose(-1, 1)

        return self.norm2(mg + y)


class SeasonalPrediction(nn.Module):
    def __init__(
        self,
        embedding_size=512,
        n_heads=8,
        dropout=0.05,
        d_layers=1,
        decomp_kernel=(32,),
        c_out=1,
        conv_kernel=(2, 4),
        isometric_kernel=(18, 6),
    ):
        super().__init__()

        self.mic = nn.ModuleList(
            [
                MIC(
                    feature_size=embedding_size,
                    n_heads=n_heads,
                    dropout=dropout,
                    decomp_kernel=decomp_kernel,
                    conv_kernel=conv_kernel,
                    isometric_kernel=isometric_kernel,
                )
                for _ in range(d_layers)
            ]
        )

        self.projection = nn.Linear(embedding_size, c_out)

    def forward(self, dec):
        for mic_layer in self.mic:
            dec = mic_layer(dec)
        return self.projection(dec)


class Model(nn.Module):
    """MICN: Multi-scale isometric convolution network for forecasting."""

    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        c_out,
        label_len=0,
        features="M",
        d_model=64,
        n_heads=4,
        d_layers=1,
        dropout=0.05,
        embed="timeF",
        freq="h",
        conv_kernel=(12, 16),
    ):
        super().__init__()
        conv_kernel = list(conv_kernel)

        decomp_kernel = []  # kernel of decomposition operation
        isometric_kernel = []  # kernel of isometric convolution
        for ii in conv_kernel:
            if ii % 2 == 0:  # the kernel of decomposition operation must be odd
                decomp_kernel.append(ii + 1)
                isometric_kernel.append((seq_len + pred_len + ii) // ii)
            else:
                decomp_kernel.append(ii)
                isometric_kernel.append((seq_len + pred_len + ii - 1) // ii)

        self.pred_len = pred_len
        self.seq_len = seq_len
        self.features = features

        # Multiple Series decomposition block from FEDformer
        self.decomp_multi = series_decomp_multi(decomp_kernel)

        # embedding
        self.dec_embedding = DataEmbedding(enc_in, d_model, embed, freq, dropout)

        self.conv_trans = SeasonalPrediction(
            embedding_size=d_model,
            n_heads=n_heads,
            dropout=dropout,
            d_layers=d_layers,
            decomp_kernel=decomp_kernel,
            c_out=c_out,
            conv_kernel=conv_kernel,
            isometric_kernel=isometric_kernel,
        )
        # refer to DLinear
        self.regression = nn.Linear(seq_len, pred_len)
        self.regression.weight = nn.Parameter(
            (1 / pred_len) * torch.ones([pred_len, seq_len]),
            requires_grad=True,
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Multi-scale Hybrid Decomposition
        seasonal_init_enc, trend = self.decomp_multi(x_enc)
        trend = self.regression(trend.permute(0, 2, 1)).permute(0, 2, 1)

        # embedding
        zeros = torch.zeros(
            [x_enc.shape[0], self.pred_len, x_enc.shape[2]],
            device=x_enc.device,
            dtype=x_enc.dtype,
        )
        seasonal_init_dec = torch.cat(
            [seasonal_init_enc[:, -self.seq_len :, :], zeros], dim=1
        )
        # The temporal mark is only added when its length matches the
        # decoder sequence (seq_len + pred_len). Upstream MICN assumes
        # label_len == seq_len; ModernTSF allows label_len == 0, in which
        # case the mark length differs and is dropped (DataEmbedding then
        # falls back to value + positional embedding only).
        if x_mark_dec is not None and x_mark_dec.shape[1] == seasonal_init_dec.shape[1]:
            dec_mark = x_mark_dec
        else:
            dec_mark = None
        dec_out = self.dec_embedding(seasonal_init_dec, dec_mark)
        dec_out = self.conv_trans(dec_out)
        dec_out = dec_out[:, -self.pred_len :, :] + trend[:, -self.pred_len :, :]
        return dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        return dec_out[:, -self.pred_len :, :]  # [B, L, D]
