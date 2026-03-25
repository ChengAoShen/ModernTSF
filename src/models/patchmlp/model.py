"""PatchMLP model implementation."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.patchmlp.layers import Emb, Encoder, SeriesDecomp


class PatchMLPModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int,
        e_layers: int,
        use_norm: bool = True,
        moving_avg: int = 13,
        patch_len: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.use_norm = use_norm
        self.decomposition = SeriesDecomp(moving_avg)
        patch_len = patch_len or [48, 24, 12, 6]
        self.emb = Emb(seq_len, d_model, patch_len)
        self.seasonal_layers = nn.ModuleList(
            [Encoder(d_model, enc_in) for _ in range(e_layers)]
        )
        self.trend_layers = nn.ModuleList(
            [Encoder(d_model, enc_in) for _ in range(e_layers)]
        )
        self.projector = nn.Linear(d_model, pred_len, bias=True)

    def forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc = x_enc / stdev

        x = x_enc.permute(0, 2, 1)
        x = self.emb(x)
        seasonal_init, trend_init = self.decomposition(x)

        for mod in self.seasonal_layers:
            seasonal_init = mod(seasonal_init)
        for mod in self.trend_layers:
            trend_init = mod(trend_init)

        x = seasonal_init + trend_init
        dec_out = self.projector(x)
        dec_out = dec_out.permute(0, 2, 1)

        if self.use_norm:
            dec_out = dec_out * (
                stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )
            dec_out = dec_out + (
                means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )

        return dec_out

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del x_mark_enc, x_dec, x_mark_dec, mask
        return self.forecast(x_enc)


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int,
        e_layers: int,
        use_norm: bool,
        moving_avg: int,
        patch_len: list[int] | None,
    ) -> None:
        super().__init__()
        self.model = PatchMLPModel(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            d_model=d_model,
            e_layers=e_layers,
            use_norm=use_norm,
            moving_avg=moving_avg,
            patch_len=patch_len,
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
