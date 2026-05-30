"""WPMixer model implementation.

Vendored/adapted from https://github.com/thuml/Time-Series-Library
(models/WPMixer.py), MIT License.

WPMixer: A Wavelet Patch Mixer for Long-Term Time Series Forecasting
(Murad et al., AAAI 2025).

Adapted for ModernTSF: the upstream ``args``-object constructor is replaced
with plain keyword arguments, and only the long-term-forecast path is kept.

The upstream relies on ``pytorch_wavelets`` (via ``pywt`` for filter
coefficients). To keep ModernTSF dependency-free we vendor a small,
self-contained 1-D DWT/IDWT (``DWT1DForward`` / ``DWT1DInverse``) that hardcodes
the orthonormal Daubechies filter banks (``db1``/``haar`` and ``db2``) and
implements the ``zero``-padding analysis/synthesis filter banks with plain
``conv1d`` / ``conv_transpose1d``. This reproduces the upstream
``layers/DWT_Decomposition.py`` behaviour for the wavelets used by the default
configs without any third-party wavelet package.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Vendored, self-contained 1-D DWT/IDWT (no pywt / pytorch_wavelets needed).
# ---------------------------------------------------------------------------

# Daubechies decomposition low-pass filters (dec_lo). The remaining three
# orthonormal filters are derived from these via the quadrature-mirror
# relations below. Values match pywt's coefficients.
_DEC_LO = {
    "db1": [0.7071067811865476, 0.7071067811865476],
    "haar": [0.7071067811865476, 0.7071067811865476],
    "db2": [
        -0.12940952255092145,
        0.22414386804185735,
        0.836516303737469,
        0.48296291314469025,
    ],
    "db3": [
        0.035226291882100656,
        -0.08544127388224149,
        -0.13501102001039084,
        0.4598775021193313,
        0.8068915093133388,
        0.3326705529509569,
    ],
}


def _wavelet_filters(name: str):
    """Return (dec_lo, dec_hi, rec_lo, rec_hi) lists for the given wavelet."""
    if name not in _DEC_LO:
        raise ValueError(
            f"Unsupported wavelet '{name}'. Vendored WPMixer DWT supports "
            f"{sorted(_DEC_LO)}."
        )
    dec_lo = list(_DEC_LO[name])
    n = len(dec_lo)
    # dec_hi[k] = (-1)^k * dec_lo[n-1-k]  (quadrature mirror)
    dec_hi = [((-1) ** k) * dec_lo[n - 1 - k] for k in range(n)]
    # Reconstruction filters are time-reversals of the decomposition filters.
    rec_lo = dec_lo[::-1]
    rec_hi = dec_hi[::-1]
    return dec_lo, dec_hi, rec_lo, rec_hi


class DWT1DForward(nn.Module):
    """Multi-level 1-D DWT forward (zero-padding), per-channel filtering.

    Matches the (yl, yh) output contract of the upstream pytorch_wavelets
    ``DWT1DForward``: yl is the final-level approximation, yh is a list of the
    detail coefficients from finest to coarsest scale.
    """

    def __init__(self, J: int = 1, wave: str = "db2", **_: object):
        super().__init__()
        self.J = J
        dec_lo, dec_hi, _, _ = _wavelet_filters(wave)
        self.filt_len = len(dec_lo)
        # conv1d cross-correlates, so reverse the filters to perform true
        # convolution (analysis filter bank).
        h0 = torch.tensor(dec_lo[::-1], dtype=torch.float).reshape(1, 1, -1)
        h1 = torch.tensor(dec_hi[::-1], dtype=torch.float).reshape(1, 1, -1)
        self.register_buffer("h0", h0)
        self.register_buffer("h1", h1)

    def _afb1d(self, x):
        # x: (B, C, L)
        B, C, L = x.shape
        Lf = self.filt_len
        # zero-padding output length (pywt 'zero' mode): floor((L + Lf - 1)/2)
        outsize = (L + Lf - 1) // 2
        p = 2 * (outsize - 1) - L + Lf
        # Prepad an extra sample after for odd total padding, like upstream.
        if p % 2 == 1:
            x = F.pad(x, (0, 1))
            p = p - 1  # remaining symmetric padding (>=0)
        pad = p // 2
        h0 = self.h0.repeat(C, 1, 1)
        h1 = self.h1.repeat(C, 1, 1)
        lo = F.conv1d(x, h0, stride=2, padding=pad, groups=C)
        hi = F.conv1d(x, h1, stride=2, padding=pad, groups=C)
        return lo, hi

    def forward(self, x):
        assert x.ndim == 3, "Can only handle 3d inputs (N, C, L)"
        highs = []
        x0 = x
        for _ in range(self.J):
            x0, x1 = self._afb1d(x0)
            highs.append(x1)
        return x0, highs


class DWT1DInverse(nn.Module):
    """Multi-level 1-D inverse DWT (zero-padding), per-channel filtering."""

    def __init__(self, wave: str = "db2", **_: object):
        super().__init__()
        _, _, rec_lo, rec_hi = _wavelet_filters(wave)
        self.filt_len = len(rec_lo)
        g0 = torch.tensor(rec_lo, dtype=torch.float).reshape(1, 1, -1)
        g1 = torch.tensor(rec_hi, dtype=torch.float).reshape(1, 1, -1)
        self.register_buffer("g0", g0)
        self.register_buffer("g1", g1)

    def _sfb1d(self, lo, hi):
        B, C, L = lo.shape
        Lf = self.filt_len
        g0 = self.g0.repeat(C, 1, 1)
        g1 = self.g1.repeat(C, 1, 1)
        # synthesis: upsample-by-2 transpose conv with padding (Lf - 2).
        pad = Lf - 2
        y = F.conv_transpose1d(lo, g0, stride=2, padding=pad, groups=C) + \
            F.conv_transpose1d(hi, g1, stride=2, padding=pad, groups=C)
        return y

    def forward(self, coeffs):
        x0, highs = coeffs
        assert x0.ndim == 3, "Can only handle 3d inputs (N, C, L)"
        for x1 in highs[::-1]:
            if x1 is None:
                x1 = torch.zeros_like(x0)
            # 'Unpad' added signal so lo and hi match in length.
            if x0.shape[-1] > x1.shape[-1]:
                x0 = x0[..., : x1.shape[-1]]
            elif x1.shape[-1] > x0.shape[-1]:
                x1 = x1[..., : x0.shape[-1]]
            x0 = self._sfb1d(x0, x1)
        return x0


class Decomposition(nn.Module):
    """Wavelet decomposition wrapper (no affine), mirrors upstream interface."""

    def __init__(self, input_length, pred_length, wavelet_name, level,
                 no_decomposition=False):
        super().__init__()
        self.input_length = input_length
        self.pred_length = pred_length
        self.level = level
        self.no_decomposition = no_decomposition

        self.dwt = DWT1DForward(wave=wavelet_name, J=level)
        self.idwt = DWT1DInverse(wave=wavelet_name)

        self.input_w_dim = (
            self._dummy_forward(self.input_length)
            if not self.no_decomposition
            else [self.input_length]
        )
        self.pred_w_dim = (
            self._dummy_forward(self.pred_length)
            if not self.no_decomposition
            else [self.pred_length]
        )

    @torch.no_grad()
    def _dummy_forward(self, input_length):
        dummy_x = torch.ones((1, 1, input_length))
        yl, yh = self.dwt(dummy_x)
        lengths = [yl.shape[-1]]
        for h in yh:
            lengths.append(h.shape[-1])
        return lengths

    def transform(self, x):
        if not self.no_decomposition:
            yl, yh = self.dwt(x)
        else:
            yl, yh = x, []
        return yl, yh

    def inv_transform(self, yl, yh):
        if not self.no_decomposition:
            x = self.idwt((yl, yh))
        else:
            x = yl
        return x


# ---------------------------------------------------------------------------
# WPMixer core (vendored from upstream models/WPMixer.py).
# ---------------------------------------------------------------------------


class TokenMixer(nn.Module):
    def __init__(self, input_seq, pred_seq, dropout, factor):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_seq, pred_seq * factor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(pred_seq * factor, pred_seq),
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.layers(x)
        x = x.transpose(1, 2)
        return x


class Mixer(nn.Module):
    def __init__(self, input_seq, out_seq, channel, d_model, dropout, tfactor,
                 dfactor):
        super().__init__()
        self.tMixer = TokenMixer(
            input_seq=input_seq, pred_seq=out_seq, dropout=dropout, factor=tfactor
        )
        self.dropoutLayer = nn.Dropout(dropout)
        self.norm1 = nn.BatchNorm2d(channel)
        self.norm2 = nn.BatchNorm2d(channel)
        self.embeddingMixer = nn.Sequential(
            nn.Linear(d_model, d_model * dfactor),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * dfactor, d_model),
        )

    def forward(self, x):
        # x: [Batch, Channel, Patch_number, d_model]
        x = self.norm1(x)
        x = x.permute(0, 3, 1, 2)
        x = self.dropoutLayer(self.tMixer(x))
        x = x.permute(0, 2, 3, 1)
        x = self.norm2(x)
        x = x + self.dropoutLayer(self.embeddingMixer(x))
        return x


class ResolutionBranch(nn.Module):
    def __init__(self, input_seq, pred_seq, channel, d_model, dropout,
                 embedding_dropout, tfactor, dfactor, patch_len, patch_stride):
        super().__init__()
        self.patch_len = patch_len
        self.patch_stride = patch_stride
        self.patch_num = int((input_seq - patch_len) / patch_stride + 2)

        self.patch_norm = nn.BatchNorm2d(channel)
        self.patch_embedding_layer = nn.Linear(patch_len, d_model)
        self.mixer1 = Mixer(
            input_seq=self.patch_num, out_seq=self.patch_num, channel=channel,
            d_model=d_model, dropout=dropout, tfactor=tfactor, dfactor=dfactor,
        )
        self.mixer2 = Mixer(
            input_seq=self.patch_num, out_seq=self.patch_num, channel=channel,
            d_model=d_model, dropout=dropout, tfactor=tfactor, dfactor=dfactor,
        )
        self.norm = nn.BatchNorm2d(channel)
        self.dropoutLayer = nn.Dropout(embedding_dropout)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2, end_dim=-1),
            nn.Linear(self.patch_num * d_model, pred_seq),
        )

    def forward(self, x):
        # x: [Batch, channel, length_of_coefficient_series]
        x_patch = self.do_patching(x)
        x_patch = self.patch_norm(x_patch)
        x_emb = self.dropoutLayer(self.patch_embedding_layer(x_patch))

        out = self.mixer1(x_emb)
        res = out
        out = res + self.mixer2(out)
        out = self.norm(out)
        out = self.head(out)
        return out

    def do_patching(self, x):
        x_end = x[:, :, -1:]
        x_padding = x_end.repeat(1, 1, self.patch_stride)
        x_new = torch.cat((x, x_padding), dim=-1)
        x_patch = x_new.unfold(
            dimension=-1, size=self.patch_len, step=self.patch_stride
        )
        return x_patch


class WPMixerCore(nn.Module):
    def __init__(self, input_length, pred_length, wavelet_name, level, channel,
                 d_model, dropout, embedding_dropout, tfactor, dfactor,
                 patch_len, patch_stride, no_decomposition):
        super().__init__()
        self.pred_length = pred_length
        self.Decomposition_model = Decomposition(
            input_length=input_length,
            pred_length=pred_length,
            wavelet_name=wavelet_name,
            level=level,
            no_decomposition=no_decomposition,
        )
        self.input_w_dim = self.Decomposition_model.input_w_dim
        self.pred_w_dim = self.Decomposition_model.pred_w_dim

        self.resolutionBranch = nn.ModuleList(
            [
                ResolutionBranch(
                    input_seq=self.input_w_dim[i],
                    pred_seq=self.pred_w_dim[i],
                    channel=channel,
                    d_model=d_model,
                    dropout=dropout,
                    embedding_dropout=embedding_dropout,
                    tfactor=tfactor,
                    dfactor=dfactor,
                    patch_len=patch_len,
                    patch_stride=patch_stride,
                )
                for i in range(len(self.input_w_dim))
            ]
        )

    def forward(self, xL):
        # xL: [Batch, look_back_length, channel]
        x = xL.transpose(1, 2)  # [batch, channel, look_back_length]
        xA, xD = self.Decomposition_model.transform(x)

        yA = self.resolutionBranch[0](xA)
        yD = []
        for i in range(len(xD)):
            yD_i = self.resolutionBranch[i + 1](xD[i])
            yD.append(yD_i)

        y = self.Decomposition_model.inv_transform(yA, yD)
        y = y.transpose(1, 2)
        xT = y[:, -self.pred_length :, :]
        return xT


class Model(nn.Module):
    """WPMixer long-term forecasting model."""

    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        label_len=0,
        features="M",
        c_out=None,
        d_model=128,
        dropout=0.1,
        tfactor=5,
        dfactor=5,
        wavelet="db2",
        level=1,
        patch_len=16,
        stride=8,
        no_decomposition=False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features
        self.c_out = c_out if c_out is not None else enc_in
        self.wpmixerCore = WPMixerCore(
            input_length=seq_len,
            pred_length=pred_len,
            wavelet_name=wavelet,
            level=level,
            channel=enc_in,
            d_model=d_model,
            dropout=dropout,
            embedding_dropout=dropout,
            tfactor=tfactor,
            dfactor=dfactor,
            patch_len=patch_len,
            patch_stride=stride,
            no_decomposition=no_decomposition,
        )

    def forecast(self, x_enc):
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        )
        x_enc = x_enc / stdev

        pred = self.wpmixerCore(x_enc)
        pred = pred[:, :, -self.c_out :]

        # De-Normalization
        dec_out = pred * (stdev[:, 0, -self.c_out :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, -self.c_out :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len :, :]  # [B, L, D]
