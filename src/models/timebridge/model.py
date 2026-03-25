"""TimeBridge model implementation."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.timebridge.layers import (
    PatchEmbed,
    ResAttention,
    TSEncoder,
    TSMixer,
    IntAttention,
    PatchSampling,
    CointAttention,
)


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        period: int,
        num_p: int | None,
        ia_layers: int,
        pd_layers: int,
        ca_layers: int,
        stable_len: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        attn_dropout: float,
        dropout: float,
        activation: str,
        revin: bool,
        time_feat_dim: int,
    ) -> None:
        super().__init__()
        self.revin = revin
        self.c_in = enc_in
        self.period = period
        self.time_feat_dim = time_feat_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_p = self.seq_len // self.period
        if num_p is None:
            num_p = self.num_p
        self.embedding = PatchEmbed(
            seq_len=self.seq_len,
            d_model=d_model,
            num_p=self.num_p,
            dropout=dropout,
        )
        layers = self._layers_init(
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            attn_dropout=attn_dropout,
            dropout=dropout,
            activation=activation,
            stable_len=stable_len,
            ia_layers=ia_layers,
            pd_layers=pd_layers,
            ca_layers=ca_layers,
            num_p=num_p,
        )
        self.encoder = TSEncoder(layers)
        out_p = self.num_p if pd_layers == 0 else num_p
        self.decoder = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(out_p * d_model, self.pred_len, bias=False),
        )

    def _layers_init(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        attn_dropout: float,
        dropout: float,
        activation: str,
        stable_len: int,
        ia_layers: int,
        pd_layers: int,
        ca_layers: int,
        num_p: int,
    ) -> list[nn.Module]:
        integrated_attention = [
            IntAttention(
                TSMixer(ResAttention(attention_dropout=attn_dropout), d_model, n_heads),
                d_model,
                d_ff,
                dropout=dropout,
                stable_len=stable_len,
                activation=activation,
                stable=True,
                enc_in=self.c_in,
            )
            for _ in range(ia_layers)
        ]
        patch_sampling = [
            PatchSampling(
                TSMixer(ResAttention(attention_dropout=attn_dropout), d_model, n_heads),
                d_model,
                d_ff,
                stable=False,
                stable_len=stable_len,
                in_p=self.num_p if i == 0 else num_p,
                out_p=num_p,
                dropout=dropout,
                activation=activation,
            )
            for i in range(pd_layers)
        ]
        cointegrated_attention = [
            CointAttention(
                TSMixer(ResAttention(attention_dropout=attn_dropout), d_model, n_heads),
                d_model,
                d_ff,
                dropout=dropout,
                activation=activation,
                stable=False,
                enc_in=self.c_in + self.time_feat_dim,
                stable_len=stable_len,
            )
            for _ in range(ca_layers)
        ]
        return [*integrated_attention, *patch_sampling, *cointegrated_attention]

    def forecast(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor | None = None):
        if x_mark_enc is None:
            x_mark_enc = torch.zeros(
                (*x_enc.shape[:-1], self.time_feat_dim), device=x_enc.device
            )
        mean = x_enc.mean(1, keepdim=True).detach()
        std = x_enc.std(1, keepdim=True).detach()
        x_enc = (x_enc - mean) / (std + 1e-5)
        x_enc = self.embedding(x_enc, x_mark_enc)
        enc_out = self.encoder(x_enc)[0][:, : self.c_in, ...]
        dec_out = self.decoder(enc_out).transpose(-1, -2)
        return dec_out * std + mean

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del x_dec, x_mark_dec, mask
        dec_out = self.forecast(x_enc, x_mark_enc)
        return dec_out[:, -self.pred_len :, :]
