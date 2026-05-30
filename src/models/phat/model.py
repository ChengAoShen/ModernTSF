"""ModernTSF adapter for the PHAT time-series forecasting model.

PHAT (https://github.com/PoorOtterBob/PHAT, ICLR 2026; arXiv:2602.00654)
is a Period Heterogeneity-Aware Transformer. It is value-only — the upstream
trainer calls ``model(x)`` with ``x`` of shape ``(B, seq_len, N)`` and the
model returns ``(B, pred_len, N)`` — so this adapter simply forwards the value
tensor and ignores the calendar marks.

The upstream repository omits the core ``PHAT_Attention`` module; it is
reconstructed from the paper in ``models.phat.layers.PHAT_Attention`` and the
rest of the model is vendored verbatim in ``models.phat._phat_model``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.phat._phat_model import PHATModel


class _PHATConfig:
    """Config object matching the attribute access in ``PHATModel``."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int,
        n_heads: int,
        d_layers: int,
        attn_dropout: float,
        ffn_dropout: float,
        ffn_expand_ratio: float,
        period_topk: int,
        period_list,
        ci: int,
        output_base_pred: int,
    ) -> None:
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_layers = d_layers
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.ffn_expand_ratio = ffn_expand_ratio
        self.period_topk = period_topk
        self.period_list = period_list
        self.CI = ci
        self.output_base_pred = output_base_pred


class Model(nn.Module):
    """Adapter wrapping the upstream PHAT model.

    Parameters
    ----------
    seq_len : int
        Input sequence length.
    pred_len : int
        Forecast horizon.
    enc_in : int
        Number of input channels (variables).
    d_model : int
        Hidden dimension.
    n_heads : int
        Number of attention heads.
    d_layers : int
        Number of PHAT transformer blocks.
    attn_dropout, ffn_dropout : float
        Dropout rates.
    ffn_expand_ratio : float
        SwiGLU FFN expansion ratio.
    period_topk : int
        Number of salient periods kept by the FFT period detector.
    period_list : list[int] | None
        Optional fixed period lengths; ``None`` uses FFT-detected periods.
    ci : int
        Channel-individual flag (1 = per-channel weights).
    output_base_pred : int
        Whether to also output the linear baseline prediction.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        d_model: int = 64,
        n_heads: int = 8,
        d_layers: int = 1,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        ffn_expand_ratio: float = 2.66667,
        period_topk: int = 1,
        period_list=None,
        ci: int = 1,
        output_base_pred: int = 0,
    ) -> None:
        super().__init__()
        config = _PHATConfig(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            d_model=d_model,
            n_heads=n_heads,
            d_layers=d_layers,
            attn_dropout=attn_dropout,
            ffn_dropout=ffn_dropout,
            ffn_expand_ratio=ffn_expand_ratio,
            period_topk=period_topk,
            period_list=period_list,
            ci=ci,
            output_base_pred=output_base_pred,
        )
        self.net = PHATModel(config)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forecast future values.

        Parameters
        ----------
        x_enc : torch.Tensor
            Input values of shape ``(B, seq_len, N)``.
        x_mark_enc, x_dec, x_mark_dec, mask
            Unused — PHAT is value-only.

        Returns
        -------
        torch.Tensor
            Forecast of shape ``(B, pred_len, N)``.
        """
        return self.net(x_enc)
