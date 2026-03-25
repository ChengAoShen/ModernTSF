"""xPatch model implementation."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.module.revin import RevIN
from models.xpatch.layers import Decomp, Network


class xPatchModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        patch_len: int,
        stride: int,
        padding_patch: str,
        ma_type: str,
        alpha: float,
        beta: float,
        revin: bool,
    ) -> None:
        super().__init__()
        self.revin = revin
        self.ma_type = ma_type
        self.revin_layer = RevIN(enc_in, affine=True, subtract_last=False)
        self.decomp = Decomp(ma_type, alpha, beta)
        self.net = Network(seq_len, pred_len, patch_len, stride, padding_patch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.revin:
            x = self.revin_layer(x, "norm")

        if self.ma_type == "reg":
            out = self.net(x, x)
        else:
            seasonal_init, trend_init = self.decomp(x)
            out = self.net(seasonal_init, trend_init)

        if self.revin:
            out = self.revin_layer(out, "denorm")
        return out


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        patch_len: int,
        stride: int,
        padding_patch: str,
        ma_type: str,
        alpha: float,
        beta: float,
        revin: bool,
    ) -> None:
        super().__init__()
        self.model = xPatchModel(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            patch_len=patch_len,
            stride=stride,
            padding_patch=padding_patch,
            ma_type=ma_type,
            alpha=alpha,
            beta=beta,
            revin=revin,
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
