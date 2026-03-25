"""SVTime model implementation."""

from __future__ import annotations

import math

import einops
import torch
from torch import nn

from models.module.revin import RevIN


def make_act(name: str):
    name = (name or "").lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "leaky_relu":
        return nn.LeakyReLU()
    return nn.Identity()


class PatchWiseAggregation(nn.Module):
    def __init__(
        self, in_period: int, out_period: int, patch_size: int, period_size: int
    ):
        super().__init__()
        self.input_period = in_period
        self.output_period = out_period
        self.period = period_size
        self.patch_size = patch_size
        self.patch_num = math.ceil(period_size / patch_size)

        weight_shape = (self.output_period, self.patch_num, self.input_period)
        self.weight = nn.Parameter(
            torch.ones(weight_shape) / self.input_period, requires_grad=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patched_x = einops.rearrange(
            x,
            "b i (p s) -> b i p s",
            p=int(self.patch_num),
            s=int(self.patch_size),
        )
        out = torch.einsum("bips,opi->bops", patched_x, self.weight)
        out = einops.rearrange(out, "b o p s -> b o (p s)")
        return out


class SVTimeModel(nn.Module):
    def __init__(
        self,
        c_in: int,
        period: int,
        seq_len: int,
        pred_len: int,
        patch_size: int,
        revin: bool = True,
        affine: bool = True,
        subtract_last: bool = False,
        analysis_act: str = "",
        analysis_hidden: str = "512,256",
    ):
        super().__init__()
        self.c_in = c_in
        self.period = period
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_size = patch_size
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        self.num_seq_periods = self.seq_len // self.period
        self.num_future_periods = (self.pred_len + self.period - 1) // self.period
        self.num_patches = math.ceil(self.period / self.patch_size)

        if isinstance(analysis_hidden, str):
            s = analysis_hidden.strip()
            hidden_sizes = [int(x) for x in s.split(",") if x] if s else []
        elif isinstance(analysis_hidden, (list, tuple)):
            hidden_sizes = [int(x) for x in analysis_hidden]
        else:
            hidden_sizes = []

        self.hidden_sizes = hidden_sizes
        self.analysis_act = analysis_act

        self.analysis_layers = self.make_analysis(hidden_sizes)

        final_size = hidden_sizes[-1] if hidden_sizes else self.num_seq_periods
        self.backcast_layer = PatchWiseAggregation(
            final_size, self.num_seq_periods, self.patch_size, self.period
        )
        self.forecast_layer = PatchWiseAggregation(
            final_size, self.num_future_periods, self.patch_size, self.period
        )

        self.trend_layer = nn.Linear(
            self.period * self.num_seq_periods, self.period * self.num_future_periods
        )

    def make_analysis(self, hidden_sizes):
        layers = []
        input_size = self.num_seq_periods
        for hidden_size in hidden_sizes:
            layers.append(
                PatchWiseAggregation(
                    input_size, hidden_size, self.patch_size, self.period
                )
            )
            layers.append(make_act(self.analysis_act))
            input_size = hidden_size
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.revin:
            x = self.revin_layer(x, "norm")
        x = x.permute(0, 2, 1)

        x = einops.rearrange(
            x,
            "b c l -> (b c) l",
        )

        if self.seq_len % self.period != 0:
            x = x[:, -(self.num_seq_periods * self.period) :]

        x_input = x.clone()

        x = einops.rearrange(
            x,
            "b (i p) -> b i p",
            p=self.period,
        )

        x = self.analysis_layers(x)

        backcast2d = self.backcast_layer(x)
        forecast2d = self.forecast_layer(x)

        backcast = einops.rearrange(backcast2d, "b i p -> b (i p)")
        forecast = einops.rearrange(forecast2d, "b i p -> b (i p)")

        trend = x_input - backcast
        trend_forecast = self.trend_layer(trend)

        x = forecast + trend_forecast
        x = x[:, : self.pred_len]

        x = einops.rearrange(
            x,
            "(b c) l -> b c l",
            c=self.c_in,
        )
        x = x.permute(0, 2, 1)

        if self.revin:
            x = self.revin_layer(x, "denorm")
        return x


class Model(nn.Module):
    def __init__(
        self,
        c_in: int,
        period: int,
        seq_len: int,
        pred_len: int,
        patch_size: int,
        revin: bool,
        affine: bool,
        subtract_last: bool,
        analysis_act: str,
        analysis_hidden: str,
    ):
        super().__init__()
        self.model = SVTimeModel(
            c_in=c_in,
            period=period,
            seq_len=seq_len,
            pred_len=pred_len,
            patch_size=patch_size,
            revin=revin,
            affine=affine,
            subtract_last=subtract_last,
            analysis_act=analysis_act,
            analysis_hidden=analysis_hidden,
        )

    def forward(self, x, *args):
        return self.model(x)
