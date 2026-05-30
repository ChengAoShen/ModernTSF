"""S4 (Structured State Space) forecasting model.

Vendored/adapted from https://github.com/state-spaces/s4
(models/s4/s4d.py and src/models/nn/dropout.py), Apache-2.0 License.

S4: Efficiently Modeling Long Sequences with Structured State Spaces
(Gu, Goel & Re, ICLR 2022).

This port uses the diagonal **S4D** variant, which is a fully pure-PyTorch
implementation (FFT-based long convolution, no custom Cauchy CUDA kernel). The
``S4DKernel`` / ``S4D`` layers below are vendored verbatim from the upstream
pedagogical ``s4d.py`` (with the ``DropoutNd`` helper inlined from
``src/models/nn/dropout.py``).

Adapted for ModernTSF: the upstream layer exposes only a sequence-to-sequence
``S4D`` block expecting ``(B, H, L)`` tensors. Here it is wrapped into a
TSLib-style forecasting ``Model`` that takes ``(B, seq_len, enc_in)`` and
returns ``(B, pred_len, c_out)``. We add per-instance normalization (from the
Non-stationary Transformer), an input projection to ``d_model``, a stack of
residual S4D blocks over the time axis, and a linear forecast head mapping
``seq_len -> pred_len``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from einops import rearrange, repeat


class DropoutNd(nn.Module):
    """Tied N-dimensional dropout (vendored from state-spaces/s4)."""

    def __init__(self, p: float = 0.5, tie: bool = True, transposed: bool = True):
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError(
                "dropout probability has to be in [0, 1), but got {}".format(p)
            )
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)

    def forward(self, X):
        """X: (batch, dim, lengths...)."""
        if self.training:
            if not self.transposed:
                X = rearrange(X, "b ... d -> b d ...")
            mask_shape = X.shape[:2] + (1,) * (X.ndim - 2) if self.tie else X.shape
            mask = torch.rand(*mask_shape, device=X.device) < 1.0 - self.p
            X = X * mask * (1.0 / (1 - self.p))
            if not self.transposed:
                X = rearrange(X, "b d ... -> b ... d")
            return X
        return X


class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters.

    Vendored from state-spaces/s4 (models/s4/s4d.py).
    """

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        H = d_model
        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N // 2))
        A_imag = math.pi * repeat(torch.arange(N // 2), "n -> h n", h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """returns: (..., c, L) where c is number of channels (default 1)."""
        # Materialize parameters
        dt = torch.exp(self.log_dt)  # (H)
        C = torch.view_as_complex(self.C)  # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device)  # (H N L)
        C = C * (torch.exp(dtA) - 1.0) / A
        K = 2 * torch.einsum("hn, hnl -> hl", C, torch.exp(K)).real
        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay."""
        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))
            optim = {"weight_decay": 0.0}
            if lr is not None:
                optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4D(nn.Module):
    """Diagonal S4 layer (vendored from state-spaces/s4, models/s4/s4d.py).

    Input and output shape (B, H, L) when ``transposed=True``.
    """

    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()
        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        self.dropout = DropoutNd(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2 * self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, **kwargs):
        """Input and output shape (B, H, L)."""
        if not self.transposed:
            u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L)  # (H L)

        # Convolution
        k_f = torch.fft.rfft(k, n=2 * L)  # (H L)
        u_f = torch.fft.rfft(u, n=2 * L)  # (B H L)
        y = torch.fft.irfft(u_f * k_f, n=2 * L)[..., :L]  # (B H L)

        # Compute D term (skip connection)
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed:
            y = y.transpose(-1, -2)
        return y, None


class S4Block(nn.Module):
    """Residual S4D block: S4D layer + dropout + residual + LayerNorm.

    Operates on ``(B, L, d_model)`` tensors (norm over the feature dim).
    """

    def __init__(self, d_model, d_state, dropout):
        super().__init__()
        self.s4d = S4D(d_model, d_state=d_state, dropout=dropout, transposed=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x: (B, L, d_model)
        z = x.transpose(-1, -2)  # (B, d_model, L)
        z, _ = self.s4d(z)
        z = z.transpose(-1, -2)  # (B, L, d_model)
        x = x + self.dropout(z)
        return self.norm(x)


class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        label_len=0,
        features="M",
        d_model=128,
        d_state=64,
        e_layers=2,
        dropout=0.1,
        use_norm=True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.label_len = label_len
        self.features = features
        self.enc_in = enc_in
        self.c_out = 1 if features == "MS" else enc_in
        self.use_norm = use_norm

        self.input_proj = nn.Linear(enc_in, d_model)
        self.blocks = nn.ModuleList(
            [S4Block(d_model, d_state, dropout) for _ in range(e_layers)]
        )
        self.time_proj = nn.Linear(seq_len, pred_len)
        self.output_proj = nn.Linear(d_model, self.c_out)

    def forecast(self, x_enc):
        # x_enc: (B, seq_len, enc_in)
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc = x_enc / stdev

        x = self.input_proj(x_enc)  # (B, seq_len, d_model)
        for block in self.blocks:
            x = block(x)

        # Map time axis seq_len -> pred_len
        x = x.transpose(-1, -2)  # (B, d_model, seq_len)
        x = self.time_proj(x)  # (B, d_model, pred_len)
        x = x.transpose(-1, -2)  # (B, pred_len, d_model)
        dec_out = self.output_proj(x)  # (B, pred_len, c_out)

        if self.use_norm:
            dec_out = dec_out * stdev[:, 0, : self.c_out].unsqueeze(1).repeat(
                1, self.pred_len, 1
            )
            dec_out = dec_out + means[:, 0, : self.c_out].unsqueeze(1).repeat(
                1, self.pred_len, 1
            )
        return dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len :, :]  # (B, pred_len, c_out)
