"""ModernTCN model implementation.

Vendored/adapted from https://github.com/luodhhh/ModernTCN
(ModernTCN-Long-term-forecasting/models/ModernTCN.py), MIT License.

ModernTCN: A Modern Pure Convolution Structure for General Time Series
Analysis (ICLR 2024). Large-kernel depthwise TCN with per-variable patch
embedding and a reparameterizable large-kernel conv block.

Adapted for ModernTSF: the upstream ``configs``-object constructor is replaced
with plain keyword arguments, only the long-term forecast path is kept
(classification / imputation / anomaly branches dropped), and the optional
time-feature embedding (``te``) branch is removed since the ModernTSF forward
contract passes temporal marks separately. The shared ``RevIN`` layer under
``models.module.revin`` is reused; the convolutional ``Block`` / ``Stage`` /
``ReparamLargeKernelConv`` and the ``series_decomp`` / ``Flatten_Head`` helpers
are ModernTCN-specific and kept local to this file.

Note: ``patch_size`` must divide ``seq_len`` (patch embedding is a strided
conv); otherwise the input is silently truncated.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.module.revin import RevIN


# ---------------------------------------------------------------------------
# Local helpers (ModernTCN_Layer.py)
# ---------------------------------------------------------------------------
class moving_avg(nn.Module):
    """Moving average block to highlight the trend of a time series."""

    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """Series decomposition block (trend / residual split)."""

    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for _ in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


# ---------------------------------------------------------------------------
# Conv building blocks (ModernTCN.py)
# ---------------------------------------------------------------------------
def get_conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    return nn.Conv1d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation,
        groups=groups,
        bias=bias,
    )


def get_bn(channels):
    return nn.BatchNorm1d(channels)


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1, bias=False):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module(
        "conv",
        get_conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        ),
    )
    result.add_module("bn", get_bn(out_channels))
    return result


class ReparamLargeKernelConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        groups,
        small_kernel,
        small_kernel_merged=False,
        nvars=7,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.small_kernel = small_kernel
        padding = kernel_size // 2
        if small_kernel_merged:
            self.lkb_reparam = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=1,
                groups=groups,
                bias=True,
            )
        else:
            self.lkb_origin = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=1,
                groups=groups,
                bias=False,
            )
            if small_kernel is not None:
                assert (
                    small_kernel <= kernel_size
                ), "The kernel size for re-param cannot be larger than the large kernel!"
                self.small_conv = conv_bn(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=small_kernel,
                    stride=stride,
                    padding=small_kernel // 2,
                    groups=groups,
                    dilation=1,
                    bias=False,
                )

    def forward(self, inputs):
        if hasattr(self, "lkb_reparam"):
            out = self.lkb_reparam(inputs)
        else:
            out = self.lkb_origin(inputs)
            if hasattr(self, "small_conv"):
                out += self.small_conv(inputs)
        return out


class Block(nn.Module):
    def __init__(self, large_size, small_size, dmodel, dff, nvars, small_kernel_merged=False, drop=0.1):
        super().__init__()
        self.dw = ReparamLargeKernelConv(
            in_channels=nvars * dmodel,
            out_channels=nvars * dmodel,
            kernel_size=large_size,
            stride=1,
            groups=nvars * dmodel,
            small_kernel=small_size,
            small_kernel_merged=small_kernel_merged,
            nvars=nvars,
        )
        self.norm = nn.BatchNorm1d(dmodel)

        # convffn1
        self.ffn1pw1 = nn.Conv1d(nvars * dmodel, nvars * dff, kernel_size=1, stride=1, padding=0, dilation=1, groups=nvars)
        self.ffn1act = nn.GELU()
        self.ffn1pw2 = nn.Conv1d(nvars * dff, nvars * dmodel, kernel_size=1, stride=1, padding=0, dilation=1, groups=nvars)
        self.ffn1drop1 = nn.Dropout(drop)
        self.ffn1drop2 = nn.Dropout(drop)

        # convffn2
        self.ffn2pw1 = nn.Conv1d(nvars * dmodel, nvars * dff, kernel_size=1, stride=1, padding=0, dilation=1, groups=dmodel)
        self.ffn2act = nn.GELU()
        self.ffn2pw2 = nn.Conv1d(nvars * dff, nvars * dmodel, kernel_size=1, stride=1, padding=0, dilation=1, groups=dmodel)
        self.ffn2drop1 = nn.Dropout(drop)
        self.ffn2drop2 = nn.Dropout(drop)

        self.ffn_ratio = dff // dmodel

    def forward(self, x):
        input = x
        B, M, D, N = x.shape
        x = x.reshape(B, M * D, N)
        x = self.dw(x)
        x = x.reshape(B, M, D, N)
        x = x.reshape(B * M, D, N)
        x = self.norm(x)
        x = x.reshape(B, M, D, N)
        x = x.reshape(B, M * D, N)

        x = self.ffn1drop1(self.ffn1pw1(x))
        x = self.ffn1act(x)
        x = self.ffn1drop2(self.ffn1pw2(x))
        x = x.reshape(B, M, D, N)

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, D * M, N)
        x = self.ffn2drop1(self.ffn2pw1(x))
        x = self.ffn2act(x)
        x = self.ffn2drop2(self.ffn2pw2(x))
        x = x.reshape(B, D, M, N)
        x = x.permute(0, 2, 1, 3)

        x = input + x
        return x


class Stage(nn.Module):
    def __init__(
        self,
        ffn_ratio,
        num_blocks,
        large_size,
        small_size,
        dmodel,
        dw_model,
        nvars,
        small_kernel_merged=False,
        drop=0.1,
    ):
        super().__init__()
        d_ffn = dmodel * ffn_ratio
        blks = []
        for _ in range(num_blocks):
            blk = Block(
                large_size=large_size,
                small_size=small_size,
                dmodel=dmodel,
                dff=d_ffn,
                nvars=nvars,
                small_kernel_merged=small_kernel_merged,
                drop=drop,
            )
            blks.append(blk)
        self.blocks = nn.ModuleList(blks)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class ModernTCN(nn.Module):
    def __init__(
        self,
        patch_size,
        patch_stride,
        stem_ratio,
        downsample_ratio,
        ffn_ratio,
        num_blocks,
        large_size,
        small_size,
        dims,
        dw_dims,
        nvars,
        small_kernel_merged=False,
        backbone_dropout=0.1,
        head_dropout=0.1,
        use_multi_scale=True,
        revin=True,
        affine=True,
        subtract_last=False,
        seq_len=512,
        c_in=7,
        individual=False,
        target_window=96,
    ):
        super().__init__()

        self.num_stage = len(num_blocks)

        # RevIN
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        # stem layer & down sampling layers (one stem + one downsample per
        # subsequent stage). Upstream hardcodes range(3) for the canonical
        # 4-stage config; here we build exactly ``num_stage`` layers so smaller
        # stage counts (e.g. a single stage) do not index past ``dims``.
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv1d(1, dims[0], kernel_size=patch_size, stride=patch_stride),
            nn.BatchNorm1d(dims[0]),
        )
        self.downsample_layers.append(stem)
        for i in range(self.num_stage - 1):
            downsample_layer = nn.Sequential(
                nn.BatchNorm1d(dims[i]),
                nn.Conv1d(dims[i], dims[i + 1], kernel_size=downsample_ratio, stride=downsample_ratio),
            )
            self.downsample_layers.append(downsample_layer)
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.downsample_ratio = downsample_ratio

        # backbone
        self.stages = nn.ModuleList()
        for stage_idx in range(self.num_stage):
            layer = Stage(
                ffn_ratio,
                num_blocks[stage_idx],
                large_size[stage_idx],
                small_size[stage_idx],
                dmodel=dims[stage_idx],
                dw_model=dw_dims[stage_idx],
                nvars=nvars,
                small_kernel_merged=small_kernel_merged,
                drop=backbone_dropout,
            )
            self.stages.append(layer)

        # head
        self.use_multi_scale = use_multi_scale
        patch_num = seq_len // patch_stride
        self.n_vars = c_in
        self.individual = individual
        d_model = dims[-1]
        if use_multi_scale:
            self.head_nf = d_model * patch_num
        else:
            if patch_num % pow(downsample_ratio, (self.num_stage - 1)) == 0:
                self.head_nf = d_model * patch_num // pow(downsample_ratio, (self.num_stage - 1))
            else:
                self.head_nf = d_model * (patch_num // pow(downsample_ratio, (self.num_stage - 1)) + 1)
        self.head = Flatten_Head(
            self.individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout
        )

    def forward_feature(self, x):
        B, M, L = x.shape
        x = x.unsqueeze(-2)
        for i in range(self.num_stage):
            B, M, D, N = x.shape
            x = x.reshape(B * M, D, N)
            if i == 0:
                if self.patch_size != self.patch_stride:
                    pad_len = self.patch_size - self.patch_stride
                    pad = x[:, :, -1:].repeat(1, 1, pad_len)
                    x = torch.cat([x, pad], dim=-1)
            else:
                if N % self.downsample_ratio != 0:
                    pad_len = self.downsample_ratio - (N % self.downsample_ratio)
                    x = torch.cat([x, x[:, :, -pad_len:]], dim=-1)
            x = self.downsample_layers[i](x)
            _, D_, N_ = x.shape
            x = x.reshape(B, M, D_, N_)
            x = self.stages[i](x)
        return x

    def forward(self, x):
        if self.revin:
            x = x.permute(0, 2, 1)
            x = self.revin_layer(x, "norm")
            x = x.permute(0, 2, 1)
        x = self.forward_feature(x)
        x = self.head(x)
        if self.revin:
            x = x.permute(0, 2, 1)
            x = self.revin_layer(x, "denorm")
            x = x.permute(0, 2, 1)
        return x


class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        features="M",
        label_len=0,
        ffn_ratio=1,
        num_blocks=(1,),
        large_size=(13,),
        small_size=(5,),
        dims=(32,),
        dw_dims=(32,),
        patch_size=16,
        patch_stride=16,
        stem_ratio=6,
        downsample_ratio=2,
        small_kernel_merged=False,
        dropout=0.1,
        head_dropout=0.1,
        use_multi_scale=True,
        revin=True,
        affine=True,
        subtract_last=False,
        individual=False,
        decomposition=False,
        kernel_size=25,
    ):
        super().__init__()
        self.features = features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.nvars = enc_in
        self.target_window = pred_len

        num_blocks = list(num_blocks)
        large_size = list(large_size)
        small_size = list(small_size)
        dims = list(dims)
        dw_dims = list(dw_dims)

        self.decomposition = decomposition
        common = dict(
            patch_size=patch_size,
            patch_stride=patch_stride,
            stem_ratio=stem_ratio,
            downsample_ratio=downsample_ratio,
            ffn_ratio=ffn_ratio,
            num_blocks=num_blocks,
            large_size=large_size,
            small_size=small_size,
            dims=dims,
            dw_dims=dw_dims,
            nvars=enc_in,
            small_kernel_merged=small_kernel_merged,
            backbone_dropout=dropout,
            head_dropout=head_dropout,
            use_multi_scale=use_multi_scale,
            revin=revin,
            affine=affine,
            subtract_last=subtract_last,
            seq_len=seq_len,
            c_in=enc_in,
            individual=individual,
            target_window=pred_len,
        )
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_res = ModernTCN(**common)
            self.model_trend = ModernTCN(**common)
        else:
            self.model = ModernTCN(**common)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        x = x_enc
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0, 2, 1)
        else:
            x = x.permute(0, 2, 1)
            x = self.model(x)
            x = x.permute(0, 2, 1)
        return x[:, -self.pred_len :, :]
