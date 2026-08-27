"""ModernTSF faithful adapter for TimeAlign (ICLR 2026).

*Bridging Past and Future: Distribution-Aware Alignment for Time Series
Forecasting.* Upstream: https://github.com/TROUBADOUR000/TimeAlign

TimeAlign is a lightweight, plug-and-play patch-MLP forecaster with a
reconstruction-based **alignment** task that bridges the past/future
distribution gap. During training it encodes both the history ``X`` and the
future ``Y``, aligns the two latent representations, and reconstructs ``Y`` from
its own encoding. The training objective is::

    L = L_pred(Ŷ, Y) + w_recon * L_recon(Y_recon, Y) + w_align * L_align

All three terms need the future ``Y``, which standard ModernTSF forwards do not
provide. This port therefore uses the trainer's opt-in conventions
(see ``benchmark.runner.trainer``):

- ``requires_train_target = True`` + ``set_train_target(y_or_none)`` — the trainer feeds the raw future
  target each training step.
- ``train_loss_override`` — the model computes the full 3-term objective in
  ``forward`` and the trainer uses it as the training loss (replacing the
  observation criterion). Validation / early-stopping still use the configured
  criterion (observation MSE/MAE).

The vendored core (``_TimeAlignCore``) reproduces the upstream ``Model`` exactly;
only the ModernTSF interface wrapper around it is new.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from components.revin import RevIN


# --------------------------------------------------------------------------- #
# Vendored layers (upstream layers/StandardNorm.py, layers/Embed.py,
# layers/Alignment.py) — copied verbatim so the package is self-contained.
# --------------------------------------------------------------------------- #
class PositionalEmbedding(nn.Module):
    """Sinusoidal positional embedding (upstream ``Embed.PositionalEmbedding``)."""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class _GlocalAlign(nn.Module):
    """Glocal alignment loss (upstream ``Alignment.glocal_align_ablation``)."""

    def __init__(self, local_margin=0.0, global_margin=0.0, loc=True, glo=True):
        super().__init__()
        self.local_margin = local_margin
        self.global_margin = global_margin
        self.loc = loc
        self.glo = glo

    @staticmethod
    def _weight_based_dynamic_loss(losses):
        n = len(losses)
        w_avg = sum(loss.detach() for loss in losses) / n
        return sum(w_avg * loss / loss.detach() for loss in losses)

    def forward(self, pred, target):
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        local_loss = torch.mean(
            F.gelu(1 - torch.abs(pred * target) - self.local_margin)
        )
        global_loss = torch.mean(
            F.gelu(
                torch.abs(
                    torch.matmul(pred, pred.transpose(1, 2))
                    - torch.matmul(target, target.transpose(1, 2))
                )
                - self.global_margin
            )
        )
        if not self.loc and not self.glo:
            return pred.new_zeros(())
        if self.loc and not self.glo:
            return local_loss
        if not self.loc and self.glo:
            return global_loss
        return self._weight_based_dynamic_loss([local_loss, global_loss])


class _PatchEmbed(nn.Module):
    """Upstream ``PatchEmbed``: non-overlapping patch projection + optional PE."""

    def __init__(self, dim, patch_len, stride=None, pos=True):
        super().__init__()
        self.patch_len = patch_len
        self.stride = patch_len if stride is None else stride
        self.patch_proj = nn.Linear(self.patch_len, dim)
        self.pos = pos
        if self.pos:
            self.pe = PositionalEmbedding(dim, 10000)

    def forward(self, x):
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = self.patch_proj(x)
        if self.pos:
            x = x + self.pe(x)
        return x


class _TimeAlignCore(nn.Module):
    """Verbatim port of the upstream TimeAlign ``Model`` (models/TimeAlign.py).

    ``forward(x, y, is_training)`` returns ``(pred, y_recon, align_loss)``.
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        patch_num,
        d_model,
        d_ff,
        e_layers,
        dropout,
        pos,
        local_margin,
        global_margin,
        loc,
        glo,
        layer_norm,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_num = patch_num
        self.d_model = d_model
        self.e_layers = e_layers

        self.patch_emb_x = _PatchEmbed(d_model, seq_len // patch_num, pos=pos)
        self.patch_emb_y = _PatchEmbed(d_model, pred_len // patch_num, pos=pos)
        self.encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model),
                )
                for _ in range(e_layers)
            ]
        )
        self.align = _GlocalAlign(local_margin, global_margin, loc, glo)
        self.ffn = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(e_layers)])
        self.autoencoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_ff),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_ff, d_model),
                )
                for _ in range(e_layers)
            ]
        )
        self.layer_norm = layer_norm
        if self.layer_norm:
            self.norm_x = nn.ModuleList(
                [nn.LayerNorm(d_model) for _ in range(e_layers)]
            )
            self.norm_y = nn.ModuleList(
                [nn.LayerNorm(d_model) for _ in range(e_layers)]
            )
        self.proj_x = nn.Linear(d_model * patch_num, pred_len)
        self.proj_y = nn.Linear(d_model * patch_num, pred_len)
        self.normalization_x = RevIN(enc_in, affine=False)
        self.normalization_y = RevIN(enc_in, affine=False)

    def forward(self, x, y, is_training=True):
        B, T, C = x.shape
        _, L, C = y.shape

        x = self.normalization_x(x, "norm")
        x = self.patch_emb_x(x.permute(0, 2, 1).reshape(-1, C * T))

        if is_training:
            y = self.normalization_y(y, "norm")
            y = self.patch_emb_y(y.permute(0, 2, 1).reshape(-1, C * L))

        align_loss = x.new_zeros(())
        for i in range(self.e_layers):
            x = x + self.encoder[i](x)
            if self.layer_norm:
                x = self.norm_x[i](x)
            if is_training:
                x_ = self.ffn[i](x)
                y = y + self.autoencoder[i](y)
                if self.layer_norm:
                    y = self.norm_y[i](y)
                align_loss = align_loss + self.align(x_, y.detach())
        align_loss = align_loss / self.e_layers

        x = self.proj_x(
            x.reshape(-1, C, self.patch_num, self.d_model).flatten(start_dim=-2)
        )
        x = x.permute(0, 2, 1)
        x = self.normalization_x(x, "denorm")

        if is_training:
            y = self.proj_y(
                y.reshape(-1, C, self.patch_num, self.d_model).flatten(start_dim=-2)
            )
            y = y.permute(0, 2, 1)
            y = self.normalization_y(y, "denorm")

        return x[:, -self.pred_len :, :], y, align_loss


class Model(nn.Module):
    """TimeAlign ModernTSF wrapper.

    Parameters
    ----------
    seq_len, pred_len, enc_in : int
        Task dimensions. ``patch_num`` must divide BOTH ``seq_len`` and
        ``pred_len``.
    patch_num, d_model, d_ff, e_layers, dropout : int/float
        Patch-MLP backbone hyper-parameters (upstream defaults d_model=32,
        d_ff=32, e_layers=2, patch_num=24 for seq_len=720).
    pos, layer_norm, loc, glo : bool
        Positional embedding / LayerNorm / local & global alignment toggles.
    local_margin, global_margin : float
        Alignment margins.
    w_recon, w_align : float
        Reconstruction and alignment loss weights (upstream defaults 1.0 / 0.1).
    """

    # Trainer convention: feed the raw future target each training step.
    requires_train_target = True

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        patch_num: int = 4,
        d_model: int = 32,
        d_ff: int = 32,
        e_layers: int = 2,
        dropout: float = 0.1,
        pos: bool = True,
        layer_norm: bool = True,
        loc: bool = True,
        glo: bool = True,
        local_margin: float = 0.0,
        global_margin: float = 0.0,
        w_recon: float = 1.0,
        w_align: float = 0.1,
    ) -> None:
        super().__init__()
        if seq_len % patch_num != 0 or pred_len % patch_num != 0:
            raise ValueError(
                f"TimeAlign requires patch_num ({patch_num}) to divide both "
                f"seq_len ({seq_len}) and pred_len ({pred_len})."
            )
        self.pred_len = pred_len
        self.w_recon = float(w_recon)
        self.w_align = float(w_align)
        self.core = _TimeAlignCore(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            patch_num=patch_num,
            d_model=d_model,
            d_ff=d_ff,
            e_layers=e_layers,
            dropout=dropout,
            pos=pos,
            local_margin=local_margin,
            global_margin=global_margin,
            loc=loc,
            glo=glo,
            layer_norm=layer_norm,
        )
        self._target: torch.Tensor | None = None
        self.train_loss_override: torch.Tensor | None = None

    def set_train_target(self, y: torch.Tensor | None) -> None:
        self._target = y

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        self.train_loss_override = None
        if self._target is not None:
            y = self._target[:, -self.pred_len :, :].float().to(x.device)
            pred, y_recon, align_loss = self.core(x, y, is_training=True)
            pred_loss = F.mse_loss(pred, y)
            recon_loss = F.mse_loss(y_recon, y)
            self.train_loss_override = (
                pred_loss + self.w_recon * recon_loss + self.w_align * align_loss
            )
            self._target = None  # consume one-shot
            return pred
        # Inference / validation: future branches are skipped; pass a zeroed
        # placeholder of the correct shape so the core can read its length.
        y_dummy = x.new_zeros((x.shape[0], self.pred_len, x.shape[2]))
        pred, _, _ = self.core(x, y_dummy, is_training=False)
        return pred
