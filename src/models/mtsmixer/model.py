"""MTSMixer model implementation.

Vendored/adapted from https://github.com/plumprc/MTS-Mixers
(models/MTSMixer.py). No license declared by the upstream repository
(no LICENSE file present).

MTS-Mixers: Multivariate Time Series Forecasting via Factorized Temporal
and Channel Mixing.

Adapted for ModernTSF: the upstream ``configs``-object constructor is
replaced with plain keyword arguments, and the shared ``RevIN`` layer under
``models.module.revin`` is reused. The factorized mixing blocks
(``MLPBlock``, ``FactorizedTemporalMixing``, ``FactorizedChannelMixing``,
``MixerBlock``) and the ``ChannelProjection`` head are MTSMixer-specific and
are vendored locally. The commented-out SVD/NMF refinement path from upstream
is dropped (forecasting-only).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.module.revin import RevIN


class MLPBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, input_dim)

    def forward(self, x):
        # [B, L, D] or [B, D, L]
        return self.fc2(self.gelu(self.fc1(x)))


class FactorizedTemporalMixing(nn.Module):
    def __init__(self, input_dim, mlp_dim, sampling):
        super().__init__()
        assert sampling in [1, 2, 3, 4, 6, 8, 12]
        self.sampling = sampling
        self.temporal_fac = nn.ModuleList(
            [MLPBlock(input_dim // sampling, mlp_dim) for _ in range(sampling)]
        )

    def merge(self, shape, x_list):
        y = torch.zeros(shape, device=x_list[0].device)
        for idx, x_pad in enumerate(x_list):
            y[:, :, idx :: self.sampling] = x_pad
        return y

    def forward(self, x):
        x_samp = []
        for idx, samp in enumerate(self.temporal_fac):
            x_samp.append(samp(x[:, :, idx :: self.sampling]))
        x = self.merge(x.shape, x_samp)
        return x


class FactorizedChannelMixing(nn.Module):
    def __init__(self, input_dim, factorized_dim):
        super().__init__()
        assert input_dim > factorized_dim
        self.channel_mixing = MLPBlock(input_dim, factorized_dim)

    def forward(self, x):
        return self.channel_mixing(x)


class MixerBlock(nn.Module):
    def __init__(
        self,
        tokens_dim,
        channels_dim,
        tokens_hidden_dim,
        channels_hidden_dim,
        fac_T,
        fac_C,
        sampling,
        norm_flag,
    ):
        super().__init__()
        self.tokens_mixing = (
            FactorizedTemporalMixing(tokens_dim, tokens_hidden_dim, sampling)
            if fac_T
            else MLPBlock(tokens_dim, tokens_hidden_dim)
        )
        self.channels_mixing = (
            FactorizedChannelMixing(channels_dim, channels_hidden_dim)
            if fac_C
            else None
        )
        self.norm = nn.LayerNorm(channels_dim) if norm_flag else None

    def forward(self, x):
        # token-mixing [B, D, #tokens]
        y = self.norm(x) if self.norm else x
        y = self.tokens_mixing(y.transpose(1, 2)).transpose(1, 2)

        # channel-mixing [B, #tokens, D]
        if self.channels_mixing:
            y = y + x
            res = y
            y = self.norm(y) if self.norm else y
            y = res + self.channels_mixing(y)

        return y


class ChannelProjection(nn.Module):
    def __init__(self, seq_len, pred_len, num_channel, individual):
        super().__init__()
        self.linears = (
            nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(num_channel)])
            if individual
            else nn.Linear(seq_len, pred_len)
        )
        self.individual = individual

    def forward(self, x):
        # x: [B, L, D]
        x_out = []
        if self.individual:
            for idx in range(x.shape[-1]):
                x_out.append(self.linears[idx](x[:, :, idx]))
            x = torch.stack(x_out, dim=-1)
        else:
            x = self.linears(x.transpose(1, 2)).transpose(1, 2)
        return x


class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        features="M",
        d_model=256,
        d_ff=64,
        e_layers=2,
        fac_T=False,
        fac_C=False,
        sampling=2,
        norm=True,
        individual=False,
        rev=True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features
        self.enc_in = enc_in

        self.mlp_blocks = nn.ModuleList(
            [
                MixerBlock(
                    seq_len,
                    enc_in,
                    d_model,
                    d_ff,
                    fac_T,
                    fac_C,
                    sampling,
                    norm,
                )
                for _ in range(e_layers)
            ]
        )
        self.norm = nn.LayerNorm(enc_in) if norm else None
        self.projection = ChannelProjection(seq_len, pred_len, enc_in, individual)
        self.rev = RevIN(enc_in) if rev else None

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x_enc: [B, seq_len, enc_in]
        x = self.rev(x_enc, "norm") if self.rev else x_enc

        for block in self.mlp_blocks:
            x = block(x)

        x = self.norm(x) if self.norm else x
        x = self.projection(x)
        x = self.rev(x, "denorm") if self.rev else x

        # [B, pred_len, c_out]
        if self.features == "MS":
            return x[:, -self.pred_len :, -1:]
        return x[:, -self.pred_len :, :]
