"""NHiTS model implementation.

Vendored/adapted from https://github.com/Nixtla/neuralforecast
(neuralforecast/models/nhits.py), Apache-2.0 License.

NHiTS: Neural Hierarchical Interpolation for Time Series Forecasting
(AAAI 2023, https://arxiv.org/abs/2201.12886). An MLP-based architecture with
hierarchical interpolation and multi-rate pooling, using backward/forward
residual links across stacked blocks.

Adapted for ModernTSF: the upstream PyTorch-Lightning ``BaseModel`` constructor
(dict-batch ``forward``) is replaced with plain keyword arguments and the
standard ``(B, T, C)`` forecasting contract. All exogenous/static branches are
dropped (forecast-only path). The core ``_IdentityBasis`` interpolation,
``NHITSBlock`` (pooling + MLP + basis), and the residual stack are vendored
locally and run channel-independently: the ``C`` channels are folded into the
batch dimension so each univariate series is forecast independently.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVATIONS = ["ReLU", "Softplus", "Tanh", "SELU", "LeakyReLU", "PReLU", "Sigmoid"]
POOLING = ["MaxPool1d", "AvgPool1d"]


class _IdentityBasis(nn.Module):
    """Hierarchical interpolation basis: backcast passthrough + forecast interp."""

    def __init__(
        self,
        backcast_size: int,
        forecast_size: int,
        interpolation_mode: str,
    ):
        super().__init__()
        assert interpolation_mode in ["linear", "nearest"]
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size
        self.interpolation_mode = interpolation_mode

    def forward(self, theta: torch.Tensor):
        backcast = theta[:, : self.backcast_size]
        knots = theta[:, self.backcast_size :]

        # [B, n_knots] -> [B, 1, n_knots]
        knots = knots.reshape(len(knots), 1, -1)
        forecast = F.interpolate(
            knots, size=self.forecast_size, mode=self.interpolation_mode
        )
        # [B, 1, H] -> [B, H]
        forecast = forecast[:, 0, :]
        return backcast, forecast


class NHITSBlock(nn.Module):
    """NHiTS block: input pooling -> MLP -> interpolation basis."""

    def __init__(
        self,
        input_size: int,
        h: int,
        n_theta: int,
        mlp_units: list,
        basis: nn.Module,
        n_pool_kernel_size: int,
        pooling_mode: str,
        dropout_prob: float,
        activation: str,
    ):
        super().__init__()

        pooled_hist_size = int(
            torch.ceil(torch.tensor(input_size / n_pool_kernel_size)).item()
        )

        assert activation in ACTIVATIONS, f"{activation} is not in {ACTIVATIONS}"
        assert pooling_mode in POOLING, f"{pooling_mode} is not in {POOLING}"

        activ = getattr(nn, activation)()

        self.pooling_layer = getattr(nn, pooling_mode)(
            kernel_size=n_pool_kernel_size, stride=n_pool_kernel_size, ceil_mode=True
        )

        # Block MLP
        hidden_layers = [nn.Linear(in_features=pooled_hist_size, out_features=mlp_units[0][0])]
        for layer in mlp_units:
            hidden_layers.append(nn.Linear(in_features=layer[0], out_features=layer[1]))
            hidden_layers.append(activ)
            if dropout_prob > 0:
                hidden_layers.append(nn.Dropout(p=dropout_prob))

        output_layer = [nn.Linear(in_features=mlp_units[-1][1], out_features=n_theta)]
        self.layers = nn.Sequential(*(hidden_layers + output_layer))
        self.basis = basis

    def forward(self, insample_y: torch.Tensor):
        # Pool1d needs 3D input (B, C, L); add channel dim
        insample_y = insample_y.unsqueeze(1)
        insample_y = self.pooling_layer(insample_y)
        insample_y = insample_y.squeeze(1)

        theta = self.layers(insample_y)
        backcast, forecast = self.basis(theta)
        return backcast, forecast


class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        features="M",
        label_len=0,
        stack_types=("identity", "identity", "identity"),
        n_blocks=(1, 1, 1),
        mlp_units=None,
        n_pool_kernel_size=(2, 2, 1),
        n_freq_downsample=(4, 2, 1),
        pooling_mode="MaxPool1d",
        interpolation_mode="linear",
        dropout=0.0,
        activation="ReLU",
        use_norm=True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.features = features
        self.use_norm = use_norm

        stack_types = list(stack_types)
        n_blocks = list(n_blocks)
        n_pool_kernel_size = list(n_pool_kernel_size)
        n_freq_downsample = list(n_freq_downsample)
        if mlp_units is None:
            mlp_units = [[256, 256]]
        # ``mlp_units`` is a list of [in, out] hidden-layer pairs shared by every
        # block (upstream NHiTS semantics). Accept a single flat pair too.
        if len(mlp_units) > 0 and isinstance(mlp_units[0], int):
            mlp_units = [list(mlp_units)]

        blocks = []
        for i in range(len(stack_types)):
            assert stack_types[i] == "identity", (
                f"Block type {stack_types[i]} not supported (only 'identity')."
            )
            for _ in range(n_blocks[i]):
                n_theta = seq_len + max(pred_len // n_freq_downsample[i], 1)
                basis = _IdentityBasis(
                    backcast_size=seq_len,
                    forecast_size=pred_len,
                    interpolation_mode=interpolation_mode,
                )
                blocks.append(
                    NHITSBlock(
                        input_size=seq_len,
                        h=pred_len,
                        n_theta=n_theta,
                        mlp_units=mlp_units,
                        basis=basis,
                        n_pool_kernel_size=n_pool_kernel_size[i],
                        pooling_mode=pooling_mode,
                        dropout_prob=dropout,
                        activation=activation,
                    )
                )
        self.blocks = nn.ModuleList(blocks)

    def _forecast(self, x_enc):
        # x_enc: [B, L, C]
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc = x_enc / stdev

        B, L, C = x_enc.shape
        # Channel-independent: fold channels into batch -> [B*C, L]
        insample_y = x_enc.permute(0, 2, 1).reshape(B * C, L)

        residuals = insample_y.flip(dims=(-1,))  # backcast init
        forecast = insample_y[:, -1:]  # Naive1 level -> [B*C, 1]
        forecast = forecast.repeat(1, self.pred_len)  # [B*C, H]
        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            residuals = residuals - backcast
            forecast = forecast + block_forecast

        # [B*C, H] -> [B, C, H] -> [B, H, C]
        dec_out = forecast.reshape(B, C, self.pred_len).permute(0, 2, 1)

        if self.use_norm:
            dec_out = dec_out * stdev.repeat(1, self.pred_len, 1)
            dec_out = dec_out + means.repeat(1, self.pred_len, 1)
        return dec_out

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self._forecast(x_enc)
        return dec_out[:, -self.pred_len :, :]  # [B, H, C]
