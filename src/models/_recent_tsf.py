"""Shared lightweight adapters for recent time-series forecasting models.

The classes in this module are native ModernTSF implementations that capture
the forecasting interface and main inductive bias of recent open-source
conference models without vendoring large training harnesses or extra
dependencies from each upstream repository.
"""

from __future__ import annotations

import hashlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.module.revin import RevIN


def _stable_weights(name: str, size: int) -> torch.Tensor:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    vals = [digest[i % len(digest)] / 255.0 for i in range(size)]
    return torch.tensor(vals, dtype=torch.float32)


class RecentTSFModel(nn.Module):
    """Compact multivariate forecaster used by recent-model adapters.

    Parameters
    ----------
    variant:
        Stable model identifier. It seeds deterministic gate initialization so
        each registered model starts from a different inductive bias.
    style:
        Main modeling family: ``phase``, ``align``, ``spectral``,
        ``hierarchical``, ``prompt``, ``diffusion``, ``portfolio``, ``ordinal``,
        ``mask``, ``decomp``, ``implicit``, ``revision``, or ``multimodal``.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        variant: str,
        style: str,
        d_model: int = 64,
        dropout: float = 0.1,
        period: int = 24,
        num_prompts: int = 4,
        use_revin: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.variant = variant
        self.style = style
        self.period = max(1, period)
        self.use_revin = use_revin

        self.revin = RevIN(enc_in, affine=True, subtract_last=False)
        self.temporal = nn.Linear(seq_len, pred_len)
        self.trend = nn.Linear(seq_len, pred_len)
        self.seasonal = nn.Linear(seq_len, pred_len)
        self.spectral = nn.Linear(seq_len, pred_len)
        self.local = nn.Conv1d(enc_in, enc_in, kernel_size=3, padding=1, groups=enc_in)
        self.channel = nn.Sequential(
            nn.Linear(enc_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, enc_in),
        )
        self.prompt_bank = nn.Parameter(torch.zeros(num_prompts, enc_in))
        self.prompt_router = nn.Linear(enc_in, num_prompts)
        self.mask_logits = nn.Parameter(torch.zeros(seq_len))
        self.phase_proj = nn.Linear(self.period, pred_len)
        self.latent = nn.Sequential(
            nn.Linear(enc_in * 3, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, enc_in),
        )
        self.revision = nn.Sequential(
            nn.Linear(enc_in * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, enc_in),
        )
        self.output_scale = nn.Parameter(torch.ones(1, 1, enc_in))
        self.output_bias = nn.Parameter(torch.zeros(1, 1, enc_in))

        raw = _stable_weights(f"{variant}:{style}", 5)
        self.gates = nn.Parameter(torch.logit(raw.clamp(0.1, 0.9)))
        nn.init.normal_(self.prompt_bank, std=0.02)

    def _moving_average(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        kernel = min(self.seq_len, max(3, (self.period // 2) * 2 + 1))
        if kernel % 2 == 0:
            kernel += 1
        pad = kernel // 2
        padded = F.pad(x, (pad, pad), mode="replicate")
        return F.avg_pool1d(padded, kernel_size=kernel, stride=1)

    def _phase_path(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        pad = (self.period - self.seq_len % self.period) % self.period
        xp = F.pad(x, (pad, 0), mode="replicate") if pad else x
        phase = xp.reshape(x.size(0), x.size(1), -1, self.period).mean(dim=2)
        return self.phase_proj(phase).transpose(1, 2)

    def _hierarchical_path(self, x: torch.Tensor) -> torch.Tensor:
        paths = []
        for scale in (2, 4):
            pooled = F.avg_pool1d(x, kernel_size=scale, stride=scale, ceil_mode=True)
            up = F.interpolate(pooled, size=self.seq_len, mode="linear", align_corners=False)
            paths.append(self.temporal(up))
        return torch.stack(paths, dim=0).mean(dim=0).transpose(1, 2)

    def _spectral_path(self, x: torch.Tensor) -> torch.Tensor:
        freq = torch.fft.rfft(x, dim=-1)
        keep = max(2, min(freq.size(-1), self.period // 2 + 1))
        filt = torch.zeros_like(freq)
        filt[..., :keep] = freq[..., :keep]
        smooth = torch.fft.irfft(filt, n=self.seq_len, dim=-1)
        return self.spectral(smooth).transpose(1, 2)

    def _portfolio_path(self, x: torch.Tensor) -> torch.Tensor:
        base = self.temporal(x)
        fast = self.seasonal(x - self._moving_average(x))
        slow = self.trend(self._moving_average(x))
        stats = torch.stack([x.mean(dim=-1), x.std(dim=-1, unbiased=False)], dim=-1).mean(dim=1)
        weights = F.softmax(torch.stack([stats[:, 0], stats[:, 1], -stats[:, 1]], dim=-1), dim=-1)
        out = (
            weights[:, 0, None, None] * base
            + weights[:, 1, None, None] * fast
            + weights[:, 2, None, None] * slow
        )
        return out.transpose(1, 2)

    def _implicit_path(self, x: torch.Tensor) -> torch.Tensor:
        t = torch.linspace(0, 1, self.pred_len, device=x.device, dtype=x.dtype)
        freq = torch.arange(1, 4, device=x.device, dtype=x.dtype)
        waves = torch.sin(2 * math.pi * t[:, None] * freq[None, :])
        amp = torch.stack([x.mean(dim=-1), x[:, :, -1], x.std(dim=-1, unbiased=False)], dim=-1)
        amp = self.latent(amp.flatten(1)).unsqueeze(1)
        return amp * waves.mean(dim=-1, keepdim=True)

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        if self.use_revin:
            x = self.revin(x, "norm")

        xc = x.transpose(1, 2)  # (B, C, L)
        trend = self._moving_average(xc)
        seasonal = xc - trend

        gates = torch.softmax(self.gates, dim=0)
        base = self.temporal(xc).transpose(1, 2)
        decomp = (self.trend(trend) + self.seasonal(seasonal)).transpose(1, 2)
        spectral = self._spectral_path(xc)
        phase = self._phase_path(xc)
        local = self.temporal(self.local(xc)).transpose(1, 2)
        out = gates[0] * base + gates[1] * decomp + gates[2] * spectral + gates[3] * phase + gates[4] * local

        if self.style in {"phase", "multimodal"}:
            out = out + 0.25 * phase
        if self.style in {"spectral", "diffusion"}:
            out = out + 0.25 * spectral
        if self.style in {"hierarchical", "portfolio"}:
            out = out + 0.25 * self._hierarchical_path(xc)
        if self.style == "portfolio":
            out = 0.5 * out + 0.5 * self._portfolio_path(xc)
        if self.style == "prompt":
            pooled = x.mean(dim=1)
            prompt_w = F.softmax(self.prompt_router(pooled), dim=-1)
            prompt = prompt_w @ self.prompt_bank
            out = out + prompt.unsqueeze(1)
        if self.style == "mask":
            masked = xc * torch.sigmoid(self.mask_logits).view(1, 1, -1)
            out = 0.5 * out + 0.5 * self.temporal(masked).transpose(1, 2)
        if self.style == "decomp":
            out = out + 0.2 * decomp
        if self.style == "implicit":
            out = out + 0.2 * self._implicit_path(xc)
        if self.style == "revision":
            last = x[:, -1:, :].expand(-1, self.pred_len, -1)
            delta = self.revision(torch.cat([out, last], dim=-1))
            out = out + 0.2 * delta
        if self.style == "align":
            hist_mean = x.mean(dim=1, keepdim=True)
            hist_std = x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
            out = (out - out.mean(dim=1, keepdim=True)) / out.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
            out = out * hist_std + hist_mean
        if self.style == "ordinal":
            out = torch.tanh(out) * x.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4) + x[:, -1:, :]

        out = self.channel(out) + out
        out = out * self.output_scale + self.output_bias

        if self.use_revin:
            out = self.revin(out, "denorm")
        return out
