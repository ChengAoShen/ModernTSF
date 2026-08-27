"""ModernTSF adapter for Glocal-IB (NeurIPS 2025), forecasting variant.

*Glocal Information Bottleneck for Time Series* — upstream:
https://github.com/Muyiiiii/NeurIPS-25-Glocal-IB

Glocal-IB is originally a **time-series imputation** regularizer: it aligns the
latent embeddings of a *masked* view and a *complete* view of a series (a
projector on the masked branch, a stop-gradient target on the complete branch),
adding ``align_weight * align_loss`` on top of the task loss. The wrapper and its
alignment losses are pure PyTorch.

Since ModernTSF is forecasting-only (no missingness), this port keeps the
**alignment mechanism faithful** and adapts the two views:

- **Anchor (complete view)** = the raw clean lookback ``x`` — the branch that
  always exists, so it produces the forecast; its embedding is the detached
  alignment target.
- **Corrupted view** = an augmented copy ``x_aug`` (random temporal masking),
  built only during training; its embedding is projected and pulled toward the
  anchor.

Training objective::

    L = L_pred(Ŷ, Y)  +  align_weight * (1 - mean cos(proj(emb_aug), emb.detach()))

The alignment term needs only ``x`` (not the future), so it rides the trainer's
existing ``aux_loss`` convention: ``self.aux_loss`` is added to the configured
prediction loss. Eval is a plain single forward, identical to the base model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from components.revin import RevIN


class _CosAlignLoss(nn.Module):
    """``1 - mean(cos)`` over the time axis (upstream ``CosAlignLoss``)."""

    def __init__(self):
        super().__init__()
        self.cos = nn.CosineSimilarity(eps=1e-8, dim=1)

    def forward(self, x_obs_p: torch.Tensor, x_ori_z: torch.Tensor) -> torch.Tensor:
        return 1.0 - self.cos(x_obs_p, x_ori_z.detach()).mean()


class _ContrastiveLoss(nn.Module):
    """InfoNCE across time steps (upstream ``ContrastiveLoss``)."""

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, x_obs_p: torch.Tensor, x_ori_z: torch.Tensor) -> torch.Tensor:
        x_obs_p = F.normalize(x_obs_p, dim=-1)
        x_ori_z = F.normalize(x_ori_z, dim=-1).detach()
        logits = torch.matmul(x_obs_p, x_ori_z.transpose(-1, -2))  # (B, T, T)
        labels = torch.arange(x_obs_p.shape[1], device=x_obs_p.device).repeat(
            logits.shape[0], 1
        )
        return self.ce(logits, labels)


_ALIGN_LOSS = {"cos_align": _CosAlignLoss, "contrastive": _ContrastiveLoss}


class _BaseForecaster(nn.Module):
    """Lightweight encoder-forecaster exposing an intermediate embedding.

    ``forward(x) -> (yhat, emb)`` with ``yhat`` of shape ``(B, pred_len, enc_in)``
    and ``emb`` of shape ``(B, seq_len, d_model)`` (the alignment latent).
    """

    def __init__(self, seq_len: int, pred_len: int, enc_in: int, d_model: int):
        super().__init__()
        self.norm = RevIN(enc_in, affine=False)
        self.encoder = nn.Sequential(
            nn.Linear(enc_in, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.temporal = nn.Linear(seq_len, pred_len)
        self.decoder = nn.Linear(d_model, enc_in)

    def forward(self, x: torch.Tensor):
        xn = self.norm(x, "norm")
        emb = self.encoder(xn)                              # (B, L, d_model)
        h = self.temporal(emb.transpose(1, 2)).transpose(1, 2)  # (B, pred_len, d_model)
        yhat = self.decoder(h)                              # (B, pred_len, enc_in)
        yhat = self.norm(yhat, "denorm")
        return yhat, emb


class Model(nn.Module):
    """Glocal-IB forecasting model.

    Parameters
    ----------
    seq_len, pred_len, enc_in : int
        Task dimensions.
    d_model : int
        Encoder / embedding width.
    align_weight : float
        Weight of the alignment regularizer (upstream default 1.0; the demo uses
        0.5).
    mask_ratio : float
        Fraction of timesteps zeroed to build the augmented (corrupted) view.
    align_loss_type : str
        ``"cos_align"`` (default, robust) or ``"contrastive"`` (InfoNCE over time).
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        align_weight: float = 0.5,
        mask_ratio: float = 0.25,
        align_loss_type: str = "cos_align",
    ) -> None:
        super().__init__()
        if align_loss_type not in _ALIGN_LOSS:
            raise ValueError(
                f"align_loss_type must be one of {sorted(_ALIGN_LOSS)}, "
                f"got {align_loss_type!r}"
            )
        self.base = _BaseForecaster(seq_len, pred_len, enc_in, d_model)
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )
        self.align_loss_fn = _ALIGN_LOSS[align_loss_type]()
        self.align_weight = float(align_weight)
        self.mask_ratio = float(mask_ratio)
        # Read by the trainer after each forward and added to the main loss.
        self.aux_loss: torch.Tensor | None = None

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """Corrupted view: zero out a random ``mask_ratio`` fraction of timesteps."""
        keep = (torch.rand(x.shape[0], x.shape[1], 1, device=x.device) >= self.mask_ratio)
        return x * keep.to(x.dtype)

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        self.aux_loss = None
        yhat, emb = self.base(x)
        # Alignment is training-only and needs no future target; gate on the
        # module's train flag (the trainer sets train()/eval() correctly).
        if self.training:
            _, emb_aug = self.base(self._augment(x))
            p = self.projection(emb_aug)
            align = self.align_loss_fn(p, emb)  # emb is detached inside the loss
            self.aux_loss = self.align_weight * align
        return yhat
