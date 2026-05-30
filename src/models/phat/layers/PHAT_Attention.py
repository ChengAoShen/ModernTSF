"""Positive-Negative X-shape Attention (PNA) for PHAT.

.. warning::

    **Unverified reconstruction.** This module is rebuilt from the paper because
    the authors never released ``PHAT_Attention``. The contractions below are
    dimensionally self-consistent and the module runs and back-propagates with
    correct shapes, but fidelity to the authors' actual implementation cannot be
    verified. PHAT results obtained with it are a best-effort approximation,
    NOT a reproduction of the paper's reported numbers.

The upstream PHAT repository (https://github.com/PoorOtterBob/PHAT) imports
``PHAT_Attention`` in ``phat/models/phat_model.py`` but never ships the file.
This module reconstructs it from the paper:

    PHAT: Modeling Period Heterogeneity for Multivariate Time Series
    Forecasting, ICLR 2026 (arXiv:2602.00654v3), Section 3.2
    "Positive-Negative X-shape Attention for Periodicity Modeling".

Integration contract (from ``phat_model.py``):
    constructor : PHAT_Attention(d_model, head, attn_dropout, layer_index)
    forward     : x of shape (B, Pb, Nb, d_model) -> (B, Pb, Nb, d_model)
where ``Pb`` is the period length (within-period offset axis) and ``Nb`` is the
number of periods (period-aligned axis). PHAT_Attention is wrapped by
``Transformer_Block`` as the attention sub-layer with a residual connection.

Equation mapping (paper -> code):
* Eq.(4)  [Q1;Q2]=ZWq, [K1;K2]=ZWk, V=ZWv, Λ=σ(ZWg)  -> ``self.to_qkv``, ``self.to_gate``
* Eq.(6)  period-offset logits ζ=µ Q1×₁K1ᵀ, η=µ Q2×₁K2ᵀ over the Pb axis
* Eq.(8)  periodic distance δ_ij = min((i-j) mod Pb, (j-i) mod Pb)
* Eq.(7)  modulated ζ̃, η̃ via positive/negative modulation terms (Softplus),
          A = softmax(ζ̃) − Λ ⊙ softmax(η̃)
* Eq.(10) period-aligned attention Ã = softmax(µ Q1×₂K1ᵀ) over the Nb axis
* Eq.(5)  PNA(Z) = A ×₁ (Ã ×₂ V)
* Eq.(11)-(12) multi-head fusion with DyT-style normalization:
          head = γ·tanh(α·(PNA(Z) + Λ⊙Z)) + β, then concat·W_O
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _periodic_distance(period: int, device: torch.device) -> torch.Tensor:
    """Periodic relative distance matrix δ (Eq. 8), shape ``(Pb, Pb)``."""
    idx = torch.arange(period, device=device)
    diff = idx[:, None] - idx[None, :]
    fwd = diff % period
    bwd = (-diff) % period
    return torch.minimum(fwd, bwd).float()


def _modulation_masks(period: int, device: torch.device):
    """Build the positive / negative modulation masks (Eq. 7).

    For target row ``m`` and column ``n`` the positive set is
    ``Δ = {s : δ[m,s] < δ[m,n]} ∪ {m}`` and the negative set is
    ``∇ = {s : δ[m,s] > δ[m,n]} ∪ {m}``. Returns two boolean tensors of shape
    ``(Pb, Pb, Pb)`` indexed ``(m, n, s)``.
    """
    delta = _periodic_distance(period, device)  # (m, s) and (m, n)
    d_ms = delta[:, None, :]  # (m, 1, s)
    d_mn = delta[:, :, None]  # (m, n, 1)
    eye = torch.eye(period, dtype=torch.bool, device=device)  # (m, s)
    self_ms = eye[:, None, :].expand(period, period, period)  # (m, n=*, s)
    pos = (d_ms < d_mn) | self_ms
    neg = (d_ms > d_mn) | self_ms
    return pos.float(), neg.float()


class PHAT_Attention(nn.Module):
    """Positive-Negative X-shape Attention (PNA) over a period bucket.

    Parameters
    ----------
    d_model : int
        Model / channel dimension of the bucket representation.
    head : int
        Number of attention heads.
    attn_dropout : float
        Dropout applied to the fused attention output.
    layer_index : int, optional
        Index of the enclosing layer (kept for signature parity with the
        upstream call ``PHAT_Attention(d_model, head, attn_dropout, l)``).
    """

    def __init__(self, d_model: int, head: int = 8, attn_dropout: float = 0.1,
                 layer_index: int = 0) -> None:
        super().__init__()
        if d_model % head != 0:
            # Fall back to a single head when d_model is not divisible.
            head = 1
        self.d_model = d_model
        self.head = head
        self.head_dim = d_model // head
        self.layer_index = layer_index
        self.scale = self.head_dim ** -0.5

        # Eq.(4): Q=[Q1;Q2], K=[K1;K2], V, and the gate Λ=σ(ZWg).
        self.to_qkv = nn.Linear(d_model, 5 * d_model)  # Q1,Q2,K1,K2,V
        self.to_gate = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(attn_dropout)

        # Eq.(10): learnable period-aligned scale µ.
        self.mu_aligned = nn.Parameter(torch.tensor(self.scale))
        # Eq.(12): per-head DyT-style normalization parameters.
        self.alpha = nn.Parameter(torch.ones(head))
        self.gamma = nn.Parameter(torch.ones(head, self.head_dim))
        self.beta = nn.Parameter(torch.zeros(head, self.head_dim))

        self._dist_cache: dict[tuple[int, torch.device], tuple] = {}

    def _masks(self, period: int, device: torch.device):
        """Cache and return the (positive, negative) modulation masks."""
        key = (period, device)
        if key not in self._dist_cache:
            self._dist_cache[key] = _modulation_masks(period, device)
        return self._dist_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply PNA to a bucket tensor of shape ``(B, Pb, Nb, d_model)``."""
        b, pb, nb, _ = x.shape
        h, hd = self.head, self.head_dim

        q1, q2, k1, k2, v = self.to_qkv(x).chunk(5, dim=-1)
        gate = torch.sigmoid(self.to_gate(x))  # (B, Pb, Nb, d_model)

        def split(t):  # (B, Pb, Nb, d) -> (B, Pb, Nb, H, hd)
            return t.reshape(b, pb, nb, h, hd)

        q1, q2, k1, k2, v = map(split, (q1, q2, k1, k2, v))
        # Per-head scalar gate Λ (mean over head_dim of the gate features).
        lam = gate.reshape(b, pb, nb, h, hd).mean(-1)  # (B, Pb, Nb, H)
        lam = lam.permute(0, 3, 1, 2)  # (B, H, Pb, Nb)

        # Eq.(6): period-offset positive/negative logits over the Pb axis.
        zeta = torch.einsum("bmnhd,bpnhd->bhmpn", q1, k1) * self.scale
        eta = torch.einsum("bmnhd,bpnhd->bhmpn", q2, k2) * self.scale

        # Eq.(7): periodic modulation, then decoupled softmax fusion.
        pos, neg = self._masks(pb, x.device)  # (m, n, s)
        zeta_t = zeta - torch.einsum("mns,bhmsk->bhmnk", pos, F.softplus(zeta))
        eta_t = eta - torch.einsum("mns,bhmsk->bhmnk", neg, F.softplus(eta))
        a_pos = torch.softmax(zeta_t, dim=3)
        a_neg = torch.softmax(eta_t, dim=3)
        a_off = a_pos - lam.unsqueeze(3) * a_neg  # (B, H, Pb, Pb, Nb)

        # Eq.(10): period-aligned attention over the Nb axis (shares Q1/K1).
        a_align = torch.softmax(
            torch.einsum("bmnhd,bmphd->bhmnp", q1, k1) * self.mu_aligned, dim=-1
        )  # (B, H, Pb, Nb, Nb)

        # Eq.(5): PNA(Z) = A ×₁ (Ã ×₂ V).
        tmp = torch.einsum("bhmnp,bmphd->bmnhd", a_align, v)
        pna = torch.einsum("bhmpn,bpnhd->bmnhd", a_off, tmp)  # (B, Pb, Nb, H, hd)

        # Eq.(12): residual gate + per-head DyT normalization.
        gated = pna + lam.permute(0, 2, 3, 1).unsqueeze(-1) * v
        out = self.gamma * torch.tanh(self.alpha[:, None] * gated) + self.beta
        out = out.reshape(b, pb, nb, self.d_model)
        return self.dropout(self.out_proj(out))



