"""CrossLinear model implementation."""

from __future__ import annotations

import math
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        seq_len: int,
        patch_num: int,
        patch_len: int,
        d_model: int,
        d_ff: int,
        variate_num: int,
    ) -> None:
        super().__init__()
        self.pad_num = patch_num * patch_len - seq_len
        self.patch_len = patch_len
        self.linear = nn.Sequential(
            nn.LayerNorm([variate_num, patch_num, patch_len]),
            nn.Linear(patch_len, d_ff),
            nn.LayerNorm([variate_num, patch_num, d_ff]),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
            nn.LayerNorm([variate_num, patch_num, d_model]),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (0, self.pad_num))
        x = x.unfold(2, self.patch_len, self.patch_len)
        return self.linear(x)


class DePatchEmbedding(nn.Module):
    def __init__(
        self,
        pred_len: int,
        patch_num: int,
        d_model: int,
        d_ff: int,
        variate_num: int,
    ) -> None:
        super().__init__()
        self.linear = nn.Sequential(
            nn.Flatten(2),
            nn.Linear(patch_num * d_model, d_ff),
            nn.LayerNorm([variate_num, d_ff]),
            nn.ReLU(),
            nn.Linear(d_ff, pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class CrossLinearModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        dec_in: int,
        patch_len: int,
        d_model: int,
        d_ff: int,
        alpha: float,
        beta: float,
    ) -> None:
        super().__init__()
        self.task_name = "long_term_forecast"
        self.ms = False
        self.eps = 1e-5
        patch_num = math.ceil(seq_len / patch_len)
        variate_num = 1 if self.ms else dec_in
        self.alpha = nn.Parameter(torch.ones([1]) * alpha)
        self.beta = nn.Parameter(torch.ones([1]) * beta)
        self.correlation_embedding = nn.Conv1d(dec_in, variate_num, 3, padding="same")
        self.value_embedding = PatchEmbedding(
            seq_len, patch_num, patch_len, d_model, d_ff, variate_num
        )
        self.pos_embedding = nn.Parameter(
            torch.randn(1, variate_num, patch_num, d_model)
        )
        if self.task_name in ("long_term_forecast", "short_term_forecast"):
            self.head = DePatchEmbedding(
                pred_len, patch_num, d_model, d_ff, variate_num
            )

    def forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        x_enc = x_enc.permute(0, 2, 1)
        x_obj = x_enc[:, [-1], :] if self.ms else x_enc
        mean = torch.mean(x_obj, dim=-1, keepdim=True)
        std = torch.std(x_obj, dim=-1, keepdim=True)
        x_enc = (x_enc - torch.mean(x_enc, dim=-1, keepdim=True)) / (
            torch.std(x_enc, dim=-1, keepdim=True) + self.eps
        )
        x_obj = x_enc[:, [-1], :] if self.ms else x_enc
        x_obj = self.alpha * x_obj + (1 - self.alpha) * self.correlation_embedding(
            x_enc
        )
        x_obj = (
            self.beta * self.value_embedding(x_obj)
            + (1 - self.beta) * self.pos_embedding
        )
        y_out = self.head(x_obj)
        y_out = y_out * std + mean
        return y_out.permute(0, 2, 1)

    def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
        if self.task_name in ("long_term_forecast", "short_term_forecast"):
            return self.forecast(x_enc)
        return x_enc


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        dec_in: int,
        patch_len: int,
        d_model: int,
        d_ff: int,
        alpha: float,
        beta: float,
    ) -> None:
        super().__init__()
        self.model = CrossLinearModel(
            seq_len=seq_len,
            pred_len=pred_len,
            dec_in=dec_in,
            patch_len=patch_len,
            d_model=d_model,
            d_ff=d_ff,
            alpha=alpha,
            beta=beta,
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del x_mark_enc, x_dec, x_mark_dec, mask
        return self.model(x_enc)
