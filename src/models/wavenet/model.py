"""WaveNet forecasting model.

Vendored/adapted from https://github.com/GestaltCogTeam/BasicTS
(baselines/WaveNet/arch.py, tag v0.5.8), Apache-2.0 License. The BasicTS
implementation is itself modified from Graph-WaveNet
(https://github.com/nnzhan/Graph-WaveNet) and ST-Norm
(https://github.com/JLDeng/ST-Norm).

Paper: WaveNet: A Generative Model for Raw Audio (https://arxiv.org/abs/1609.03499).

Adapted for ModernTSF:
- The upstream BasicTS forward contract ``forward(history_data (B, L, N, C), ...)``
  is replaced with the ModernTSF ``forward(x_enc (B, T, C), ...)`` contract that
  returns ``(B, pred_len, C)``.
- The core dilated causal-convolution stack (gated tanh/sigmoid activations,
  residual + skip connections, exponentially growing dilation) is kept intact.
  Each ModernTSF channel is mapped onto the ``N`` (node) axis and the single
  input feature onto ``in_dim``.
- A RevIN-style instance normalization (reused from ``models.module.revin``) is
  applied around the network for stable long-horizon forecasting.
- Non-forecasting task branches are dropped.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.module.revin import RevIN


class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        label_len=0,
        features="M",
        residual_channels=16,
        dilation_channels=16,
        skip_channels=64,
        end_channels=128,
        kernel_size=2,
        blocks=2,
        layers=2,
        use_norm=True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.features = features
        self.blocks = blocks
        self.layers = layers
        self.use_norm = use_norm

        if use_norm:
            self.revin = RevIN(enc_in, affine=True)

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()

        # single feature channel per node (the value at each timestep)
        self.start_conv = nn.Conv2d(
            in_channels=1, out_channels=residual_channels, kernel_size=(1, 1)
        )

        receptive_field = 1
        for _ in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for _ in range(layers):
                self.filter_convs.append(
                    nn.Conv2d(
                        residual_channels,
                        dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=new_dilation,
                    )
                )
                self.gate_convs.append(
                    nn.Conv2d(
                        residual_channels,
                        dilation_channels,
                        kernel_size=(1, kernel_size),
                        dilation=new_dilation,
                    )
                )
                self.residual_convs.append(
                    nn.Conv2d(dilation_channels, residual_channels, kernel_size=(1, 1))
                )
                self.skip_convs.append(
                    nn.Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1))
                )
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

        self.end_conv_1 = nn.Conv2d(
            skip_channels, end_channels, kernel_size=(1, 1), bias=True
        )
        # map the temporal skip summary onto the prediction horizon
        self.end_conv_2 = nn.Conv2d(
            end_channels, pred_len, kernel_size=(1, 1), bias=True
        )

        self.receptive_field = receptive_field

    def forecast(self, x_enc):
        # x_enc: [B, T, C]
        if self.use_norm:
            x_enc = self.revin(x_enc, "norm")

        # -> [B, in_dim=1, N=C, L=T]
        x = x_enc.permute(0, 2, 1).unsqueeze(1)

        in_len = x.size(3)
        if in_len < self.receptive_field:
            x = F.pad(x, (self.receptive_field - in_len, 0, 0, 0))

        x = self.start_conv(x)
        skip = 0

        for i in range(self.blocks * self.layers):
            residual = x
            filt = torch.tanh(self.filter_convs[i](residual))
            gate = torch.sigmoid(self.gate_convs[i](residual))
            x = filt * gate

            s = self.skip_convs[i](x)
            if isinstance(skip, torch.Tensor):
                skip = skip[:, :, :, -s.size(3):]
            skip = s + skip

            x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)

        # collapse remaining temporal width into a single summary column
        x = F.relu(skip)
        x = x.mean(dim=3, keepdim=True)  # [B, skip_channels, N, 1]
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x)  # [B, pred_len, N, 1]

        # -> [B, pred_len, C]
        dec_out = x.squeeze(-1).contiguous()

        if self.use_norm:
            dec_out = self.revin(dec_out, "denorm")
        return dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len:, :]
