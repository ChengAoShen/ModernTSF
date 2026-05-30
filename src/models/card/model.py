"""CARD model implementation.

Vendored/adapted from https://github.com/wxie9/CARD
(long_term_forecast_l96/models/CARD.py). The upstream repository ships no
explicit LICENSE file; it is built on top of the Time-Series-Library (TSLib,
MIT License) framework, from which the experiment harness and layer
conventions are inherited.

CARD: A Channel Aligned Robust Blend Transformer for Time Series Forecasting
(ICLR 2024). The architecture applies dual attention -- over patch tokens
(with an exponential-moving-average smoothing kernel) and over the channel /
hidden dimension (with low-rank dynamic projection) -- and a token-blend
merge across attention heads.

Adapted for ModernTSF: the upstream ``config``-object constructor is replaced
with plain keyword arguments, ``total_token_number`` is computed internally
instead of mutating a shared config object, and only the long-term forecast
path is kept (classification / imputation / anomaly branches dropped). The
custom dual attention has no equivalent in ``models.module.*`` and is kept
local to this file. The forward signature follows the ModernTSF
``(x_enc, x_mark_enc, x_dec, x_mark_dec)`` contract and returns
``(B, pred_len, c_out)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


class Attenion(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        enc_in,
        d_ff,
        dropout,
        dp_rank,
        merge_size,
        momentum,
        alpha,
        total_token_number,
        over_hidden=False,
    ):
        super().__init__()

        self.over_hidden = over_hidden
        self.n_heads = n_heads
        self.c_in = enc_in
        self.qkv = nn.Linear(d_model, d_model * 3, bias=True)

        self.attn_dropout = nn.Dropout(dropout)
        self.head_dim = d_model // n_heads

        self.dropout_mlp = nn.Dropout(dropout)
        self.mlp = nn.Linear(d_model, d_model)

        self.norm_post1 = nn.Sequential(
            Transpose(1, 2), nn.BatchNorm1d(d_model, momentum=momentum), Transpose(1, 2)
        )
        self.norm_post2 = nn.Sequential(
            Transpose(1, 2), nn.BatchNorm1d(d_model, momentum=momentum), Transpose(1, 2)
        )
        self.norm_attn = nn.Sequential(
            Transpose(1, 2), nn.BatchNorm1d(d_model, momentum=momentum), Transpose(1, 2)
        )

        self.dp_rank = dp_rank
        self.dp_k = nn.Linear(self.head_dim, self.dp_rank)
        self.dp_v = nn.Linear(self.head_dim, self.dp_rank)

        self.ff_1 = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=True),
        )
        self.ff_2 = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=True),
        )
        self.merge_size = merge_size

        ema_size = max(enc_in, total_token_number, dp_rank)
        ema_matrix = torch.zeros((ema_size, ema_size))
        ema_matrix[0][0] = 1
        for i in range(1, total_token_number):
            for j in range(i):
                ema_matrix[i][j] = ema_matrix[i - 1][j] * (1 - alpha)
            ema_matrix[i][i] = alpha
        self.register_buffer("ema_matrix", ema_matrix)

    def ema(self, src):
        return torch.einsum(
            "bnhad,ga ->bnhgd",
            src,
            self.ema_matrix[: src.shape[-2], : src.shape[-2]],
        )

    def dynamic_projection(self, src, mlp):
        src_dp = mlp(src)
        src_dp = F.softmax(src_dp, dim=-1)
        src_dp = torch.einsum("bnhef,bnhec -> bnhcf", src, src_dp)
        return src_dp

    def forward(self, src, *args, **kwargs):
        B, nvars, H, C = src.shape

        qkv = (
            self.qkv(src)
            .reshape(B, nvars, H, 3, self.n_heads, C // self.n_heads)
            .permute(3, 0, 1, 4, 2, 5)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        if not self.over_hidden:
            attn_score_along_token = (
                torch.einsum("bnhed,bnhfd->bnhef", self.ema(q), self.ema(k))
                / self.head_dim**-0.5
            )
            attn_along_token = self.attn_dropout(
                F.softmax(attn_score_along_token, dim=-1)
            )
            output_along_token = torch.einsum(
                "bnhef,bnhfd->bnhed", attn_along_token, v
            )
        else:
            v_dp, k_dp = (
                self.dynamic_projection(v, self.dp_v),
                self.dynamic_projection(k, self.dp_k),
            )
            attn_score_along_token = (
                torch.einsum("bnhed,bnhfd->bnhef", self.ema(q), self.ema(k_dp))
                / self.head_dim**-0.5
            )
            attn_along_token = self.attn_dropout(
                F.softmax(attn_score_along_token, dim=-1)
            )
            output_along_token = torch.einsum(
                "bnhef,bnhfd->bnhed", attn_along_token, v_dp
            )

        attn_score_along_hidden = (
            torch.einsum("bnhae,bnhaf->bnhef", q, k) / q.shape[-2] ** -0.5
        )
        attn_along_hidden = self.attn_dropout(
            F.softmax(attn_score_along_hidden, dim=-1)
        )
        output_along_hidden = torch.einsum(
            "bnhef,bnhaf->bnhae", attn_along_hidden, v
        )

        merge_size = self.merge_size
        output1 = rearrange(
            output_along_token.reshape(B * nvars, -1, self.head_dim),
            "bn (hl1 hl2 hl3) d -> bn  hl2 (hl3 hl1) d",
            hl1=self.n_heads // merge_size,
            hl2=output_along_token.shape[-2],
            hl3=merge_size,
        ).reshape(B * nvars, -1, self.head_dim * self.n_heads)

        output2 = rearrange(
            output_along_hidden.reshape(B * nvars, -1, self.head_dim),
            "bn (hl1 hl2 hl3) d -> bn  hl2 (hl3 hl1) d",
            hl1=self.n_heads // merge_size,
            hl2=output_along_token.shape[-2],
            hl3=merge_size,
        ).reshape(B * nvars, -1, self.head_dim * self.n_heads)

        output1 = self.norm_post1(output1)
        output1 = output1.reshape(B, nvars, -1, self.n_heads * self.head_dim)
        output2 = self.norm_post2(output2)
        output2 = output2.reshape(B, nvars, -1, self.n_heads * self.head_dim)

        src2 = self.ff_1(output1) + self.ff_2(output2)

        src = src + src2
        src = src.reshape(B * nvars, -1, self.n_heads * self.head_dim)
        src = self.norm_attn(src)
        src = src.reshape(B, nvars, -1, self.n_heads * self.head_dim)
        return src


class CARDformer(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        patch_len,
        stride,
        d_model,
        n_heads,
        e_layers,
        d_ff,
        dropout,
        dp_rank,
        merge_size,
        momentum,
        alpha,
        use_statistic,
    ):
        super().__init__()

        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        patch_num = int((seq_len - self.patch_len) / self.stride + 1)
        self.patch_num = patch_num
        self.W_pos_embed = nn.Parameter(torch.randn(patch_num, d_model) * 1e-2)
        self.model_token_number = 0

        self.total_token_number = self.patch_num + self.model_token_number + 1

        self.W_input_projection = nn.Linear(self.patch_len, d_model)
        self.input_dropout = nn.Dropout(dropout)

        self.use_statistic = use_statistic
        self.W_statistic = nn.Linear(2, d_model)
        self.cls = nn.Parameter(torch.randn(1, d_model) * 1e-2)

        self.W_out = nn.Linear(
            (patch_num + 1 + self.model_token_number) * d_model, pred_len
        )

        def make_attn(over_hidden):
            return Attenion(
                d_model=d_model,
                n_heads=n_heads,
                enc_in=enc_in,
                d_ff=d_ff,
                dropout=dropout,
                dp_rank=dp_rank,
                merge_size=merge_size,
                momentum=momentum,
                alpha=alpha,
                total_token_number=self.total_token_number,
                over_hidden=over_hidden,
            )

        self.Attentions_over_token = nn.ModuleList(
            [make_attn(False) for _ in range(e_layers)]
        )
        self.Attentions_over_channel = nn.ModuleList(
            [make_attn(True) for _ in range(e_layers)]
        )
        self.Attentions_mlp = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(e_layers)]
        )
        self.Attentions_dropout = nn.ModuleList(
            [nn.Dropout(dropout) for _ in range(e_layers)]
        )
        self.Attentions_norm = nn.ModuleList(
            [
                nn.Sequential(
                    Transpose(1, 2),
                    nn.BatchNorm1d(d_model, momentum=momentum),
                    Transpose(1, 2),
                )
                for _ in range(e_layers)
            ]
        )

    def forward(self, z):
        # z: [B, C, S]
        z_mean = torch.mean(z, dim=(-1), keepdims=True)
        z_std = torch.std(z, dim=(-1), keepdims=True)
        z = (z - z_mean) / (z_std + 1e-4)

        zcube = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        z_embed = self.input_dropout(self.W_input_projection(zcube)) + self.W_pos_embed

        if self.use_statistic:
            z_stat = torch.cat((z_mean, z_std), dim=-1)
            if z_stat.shape[-2] > 1:
                z_stat = (z_stat - torch.mean(z_stat, dim=-2, keepdims=True)) / (
                    torch.std(z_stat, dim=-2, keepdims=True) + 1e-4
                )
            z_stat = self.W_statistic(z_stat)
            z_embed = torch.cat((z_stat.unsqueeze(-2), z_embed), dim=-2)
        else:
            cls_token = self.cls.repeat(z_embed.shape[0], z_embed.shape[1], 1, 1)
            z_embed = torch.cat((cls_token, z_embed), dim=-2)

        inputs = z_embed
        b, c, t, h = inputs.shape
        for a_2, a_1, mlp, drop, norm in zip(
            self.Attentions_over_token,
            self.Attentions_over_channel,
            self.Attentions_mlp,
            self.Attentions_dropout,
            self.Attentions_norm,
        ):
            output_1 = a_1(inputs.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
            output_2 = a_2(output_1)
            outputs = drop(mlp(output_1 + output_2)) + inputs
            outputs = norm(outputs.reshape(b * c, t, -1)).reshape(b, c, t, -1)
            inputs = outputs

        z_out = self.W_out(outputs.reshape(b, c, -1))
        z = z_out * (z_std + 1e-4) + z_mean
        return z


class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        features="M",
        patch_len=16,
        stride=8,
        d_model=128,
        n_heads=8,
        e_layers=2,
        d_ff=256,
        dropout=0.1,
        dp_rank=8,
        merge_size=2,
        momentum=0.1,
        alpha=0.5,
        use_statistic=False,
    ):
        super().__init__()
        self.features = features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.model = CARDformer(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_heads=n_heads,
            e_layers=e_layers,
            d_ff=d_ff,
            dropout=dropout,
            dp_rank=dp_rank,
            merge_size=merge_size,
            momentum=momentum,
            alpha=alpha,
            use_statistic=use_statistic,
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x_enc: [B, seq_len, C]
        x = x_enc.permute(0, 2, 1)  # [B, C, seq_len]
        x = self.model(x)  # [B, C, pred_len]
        x = x.permute(0, 2, 1)  # [B, pred_len, C]
        return x[:, -self.pred_len :, :]
