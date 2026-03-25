"""TimeBridge layers."""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PatchEmbed(nn.Module):
    def __init__(self, seq_len: int, d_model: int, num_p: int, dropout: float) -> None:
        super().__init__()
        self.num_p = num_p
        self.patch = seq_len // self.num_p
        self.proj = nn.Sequential(
            nn.Linear(self.patch, d_model, False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, x_mark: torch.Tensor) -> torch.Tensor:
        total_len = self.num_p * self.patch
        seq_len = x.shape[1]
        if seq_len > total_len:
            x = x[:, -total_len:, :]
            x_mark = x_mark[:, -total_len:, :]
        elif seq_len < total_len:
            pad_len = total_len - seq_len
            x = F.pad(x, (0, 0, pad_len, 0))
            x_mark = F.pad(x_mark, (0, 0, pad_len, 0))
        x = torch.cat([x, x_mark], dim=-1).transpose(-1, -2)
        x = x.reshape(*x.shape[:-1], self.num_p, self.patch)
        return self.proj(x)


class TSEncoder(nn.Module):
    def __init__(self, attn_layers: list[nn.Module]) -> None:
        super().__init__()
        self.attn_layers = nn.ModuleList(attn_layers)

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)
        return x, attns


def period_norm(x: torch.Tensor, period_len: int) -> torch.Tensor:
    if len(x.shape) == 3:
        x = x.unsqueeze(-2)
    b, c, n, t = x.shape
    x_patch = [x[..., period_len - 1 - i : -i + t] for i in range(0, period_len)]
    x_patch = torch.stack(x_patch, dim=-1)
    mean = x_patch.mean(4)
    mean = F.pad(
        mean.reshape(b * c, n, -1), mode="replicate", pad=(period_len - 1, 0)
    ).reshape(b, c, n, -1)
    out = x - mean
    return out.squeeze(-2)


class ResAttention(nn.Module):
    def __init__(self, attention_dropout: float = 0.1, scale=None) -> None:
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, res=False, attn=None):
        b, l, h, e = queries.shape
        _, s, _, _ = values.shape
        scale = self.scale or 1.0 / math.sqrt(e)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        attn_map = torch.softmax(scale * scores, dim=-1)
        a = self.dropout(attn_map)
        v = torch.einsum("bhls,bshd->blhd", a, values)
        return v.contiguous(), a


class TSMixer(nn.Module):
    def __init__(self, attention: nn.Module, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.attention = attention
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.n_heads = n_heads

    def forward(self, q, k, v, res=False, attn=None):
        b, l, _ = q.shape
        _, s, _ = k.shape
        h = self.n_heads
        q = self.q_proj(q).reshape(b, l, h, -1)
        k = self.k_proj(k).reshape(b, s, h, -1)
        v = self.v_proj(v).reshape(b, s, h, -1)
        out, attn = self.attention(q, k, v, res=res, attn=attn)
        out = out.view(b, l, -1)
        return self.out(out), attn


class IntAttention(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        d_ff: int | None = None,
        stable_len: int = 8,
        dropout: float = 0.1,
        activation: str = "relu",
        stable: bool = True,
        enc_in: int | None = None,
    ) -> None:
        super().__init__()
        self.stable = stable
        self.stable_len = stable_len
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x = self.temporal_attn(x)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.fc1(y)))
        y = self.dropout(self.fc2(y))
        return self.norm2(x + y), None

    def temporal_attn(self, x: torch.Tensor) -> torch.Tensor:
        b, c, n, d = x.shape
        new_x = x.reshape(-1, n, d)
        qk = new_x
        if self.stable:
            with torch.no_grad():
                qk = period_norm(new_x, self.stable_len)
        new_x = self.attention(qk, qk, new_x)[0]
        return new_x.reshape(b, c, n, d)


class PatchSampling(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
        activation: str = "relu",
        in_p: int = 30,
        out_p: int = 4,
        stable: bool = False,
        stable_len: int = 8,
    ) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.in_p = in_p
        self.out_p = out_p
        self.stable = stable
        self.stable_len = stable_len
        self.attention = attention
        self.conv1 = nn.Conv1d(self.in_p, self.out_p, 1, 1, 0, bias=False)
        self.conv2 = nn.Conv1d(self.out_p + 1, self.out_p, 1, 1, 0, bias=False)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x = self.down_attn(x)
        y = x = self.norm1(new_x)
        y = self.dropout(self.activation(self.fc1(y)))
        y = self.dropout(self.fc2(y))
        return self.norm2(x + y), None

    def down_attn(self, x: torch.Tensor) -> torch.Tensor:
        b, c, n, d = x.shape
        x = x.reshape(-1, n, d)
        new_x = self.conv1(x)
        new_x = self.conv2(torch.cat([new_x, x.mean(-2, keepdim=True)], dim=-2)) + new_x
        new_x = self.attention(new_x, x, x)[0] + self.dropout(new_x)
        return new_x.reshape(b, c, -1, d)


class CointAttention(nn.Module):
    def __init__(
        self,
        attention: nn.Module,
        d_model: int,
        d_ff: int | None = None,
        axial: bool = True,
        stable_len: int = 8,
        dropout: float = 0.1,
        activation: str = "relu",
        stable: bool = True,
        enc_in: int | None = None,
    ) -> None:
        super().__init__()
        self.stable = stable
        self.stable_len = stable_len
        d_ff = d_ff or 4 * d_model
        self.axial_func = axial
        self.attention1 = attention
        self.attention2 = copy.deepcopy(attention)
        enc_in = enc_in or 1
        self.num_rc = math.ceil(enc_in**0.5)
        self.pad_ch = nn.ConstantPad1d((0, self.num_rc**2 - enc_in), 0)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        if self.axial_func is True:
            new_x = self.axial_attn(x)
        else:
            new_x = self.full_attn(x)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.fc1(y)))
        y = self.dropout(self.fc2(y))
        return self.norm2(x + y), None

    def axial_attn(self, x: torch.Tensor) -> torch.Tensor:
        b, c, n, d = x.shape
        new_x = rearrange(x, "b c n d -> (b n) c d")
        new_x = (
            self.pad_ch(new_x.transpose(-1, -2))
            .transpose(-1, -2)
            .reshape(-1, self.num_rc, d)
        )
        new_x = self.attention1(new_x, new_x, new_x)[0]
        new_x = rearrange(new_x, "(b r) c d -> (b c) r d", r=self.num_rc)
        new_x = self.attention2(new_x, new_x, new_x)[0] + new_x
        new_x = rearrange(new_x, "(b n c) r d -> b (r c) n d", b=b, n=n)
        return new_x[:, :c, ...]

    def full_attn(self, x: torch.Tensor) -> torch.Tensor:
        b, c, n, d = x.shape
        new_x = rearrange(x, "b c n d -> (b n) c d")
        new_x = self.attention1(new_x, new_x, new_x)[0]
        new_x = rearrange(new_x, "(b n) c d -> b c n d", b=b, n=n)
        return new_x[:, :c, :]
