"""Koopa model implementation.

Vendored/adapted from https://github.com/thuml/Time-Series-Library
(models/Koopa.py), MIT License.

Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors
(NeurIPS 2023).

Adapted for ModernTSF: the upstream ``configs``-object constructor is replaced
with plain keyword arguments, and only the long-term forecasting path is kept
(classification / imputation / anomaly branches dropped). Upstream computes the
shared Fourier ``mask_spectrum`` from the full training set at construction
time; since the ModernTSF model factory has no data access, the mask is instead
estimated lazily from the first forward batch (a per-channel top-alpha frequency
mask). The Koopman blocks (``FourierFilter``, ``MLP``, ``KPLayer``,
``KPLayerApprox``, ``TimeVarKP``, ``TimeInvKP``) are Koopa-specific and kept
local to this file.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class FourierFilter(nn.Module):
    """Split a series into time-variant and time-invariant terms."""

    def __init__(self, mask_spectrum):
        super().__init__()
        self.mask_spectrum = mask_spectrum

    def forward(self, x):
        xf = torch.fft.rfft(x, dim=1)
        mask = torch.ones_like(xf)
        mask[:, self.mask_spectrum, :] = 0
        x_var = torch.fft.irfft(xf * mask, dim=1, n=x.shape[1])
        x_inv = x - x_var
        return x_var, x_inv


class MLP(nn.Module):
    """Encode/decode high-dimension representation of sequential data."""

    def __init__(
        self,
        f_in,
        f_out,
        hidden_dim=128,
        hidden_layers=2,
        dropout=0.05,
        activation="tanh",
    ):
        super().__init__()
        self.f_in = f_in
        self.f_out = f_out
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            raise NotImplementedError

        layers = [
            nn.Linear(self.f_in, self.hidden_dim),
            self.activation,
            nn.Dropout(self.dropout),
        ]
        for _ in range(self.hidden_layers - 2):
            layers += [
                nn.Linear(self.hidden_dim, self.hidden_dim),
                self.activation,
                nn.Dropout(dropout),
            ]

        layers += [nn.Linear(hidden_dim, f_out)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        # x: B x S x f_in -> y: B x S x f_out
        return self.layers(x)


class KPLayer(nn.Module):
    """One-step transition of a linear system found by DMD iteratively."""

    def __init__(self):
        super().__init__()
        self.K = None  # B E E

    def one_step_forward(self, z, return_rec=False, return_K=False):
        B, input_len, E = z.shape
        assert input_len > 1, "snapshots number should be larger than 1"
        x, y = z[:, :-1], z[:, 1:]

        # solve linear system
        self.K = torch.linalg.lstsq(x, y).solution  # B E E
        if torch.isnan(self.K).any():
            print("Encounter K with nan, replace K by identity matrix")
            self.K = (
                torch.eye(self.K.shape[1])
                .to(self.K.device)
                .unsqueeze(0)
                .repeat(B, 1, 1)
            )

        z_pred = torch.bmm(z[:, -1:], self.K)
        if return_rec:
            z_rec = torch.cat((z[:, :1], torch.bmm(x, self.K)), dim=1)
            return z_rec, z_pred

        return z_pred

    def forward(self, z, pred_len=1):
        assert pred_len >= 1, "prediction length should not be less than 1"
        z_rec, z_pred = self.one_step_forward(z, return_rec=True)
        z_preds = [z_pred]
        for _ in range(1, pred_len):
            z_pred = torch.bmm(z_pred, self.K)
            z_preds.append(z_pred)
        z_preds = torch.cat(z_preds, dim=1)
        return z_rec, z_preds


class KPLayerApprox(nn.Module):
    """Koopman transition with multistep K approximation."""

    def __init__(self):
        super().__init__()
        self.K = None  # B E E
        self.K_step = None  # B E E

    def forward(self, z, pred_len=1):
        # z: B L E koopman invariance space representation
        B, input_len, E = z.shape
        assert input_len > 1, "snapshots number should be larger than 1"
        x, y = z[:, :-1], z[:, 1:]

        # solve linear system
        self.K = torch.linalg.lstsq(x, y).solution  # B E E

        if torch.isnan(self.K).any():
            print("Encounter K with nan, replace K by identity matrix")
            self.K = (
                torch.eye(self.K.shape[1])
                .to(self.K.device)
                .unsqueeze(0)
                .repeat(B, 1, 1)
            )

        z_rec = torch.cat((z[:, :1], torch.bmm(x, self.K)), dim=1)  # B L E

        if pred_len <= input_len:
            self.K_step = torch.linalg.matrix_power(self.K, pred_len)
            if torch.isnan(self.K_step).any():
                print(
                    "Encounter multistep K with nan, replace it by identity matrix"
                )
                self.K_step = (
                    torch.eye(self.K_step.shape[1])
                    .to(self.K_step.device)
                    .unsqueeze(0)
                    .repeat(B, 1, 1)
                )
            z_pred = torch.bmm(z[:, -pred_len:, :], self.K_step)
        else:
            self.K_step = torch.linalg.matrix_power(self.K, input_len)
            if torch.isnan(self.K_step).any():
                print(
                    "Encounter multistep K with nan, replace it by identity matrix"
                )
                self.K_step = (
                    torch.eye(self.K_step.shape[1])
                    .to(self.K_step.device)
                    .unsqueeze(0)
                    .repeat(B, 1, 1)
                )
            temp_z_pred, all_pred = z, []
            for _ in range(math.ceil(pred_len / input_len)):
                temp_z_pred = torch.bmm(temp_z_pred, self.K_step)
                all_pred.append(temp_z_pred)
            z_pred = torch.cat(all_pred, dim=1)[:, :pred_len, :]

        return z_rec, z_pred


class TimeVarKP(nn.Module):
    """Koopman predictor with DMD (analytical Koopman operator).

    Uses local variations within individual sliding windows to predict the
    future of the time-variant term.
    """

    def __init__(
        self,
        enc_in=8,
        input_len=96,
        pred_len=96,
        seg_len=24,
        dynamic_dim=128,
        encoder=None,
        decoder=None,
        multistep=False,
    ):
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.seg_len = seg_len
        self.dynamic_dim = dynamic_dim
        self.multistep = multistep
        self.encoder, self.decoder = encoder, decoder
        self.freq = math.ceil(self.input_len / self.seg_len)  # input seg number
        self.step = math.ceil(self.pred_len / self.seg_len)  # output seg number
        self.padding_len = self.seg_len * self.freq - self.input_len
        self.dynamics = KPLayerApprox() if self.multistep else KPLayer()

    def forward(self, x):
        # x: B L C
        B, L, C = x.shape

        res = torch.cat((x[:, L - self.padding_len :, :], x), dim=1)

        res = res.chunk(self.freq, dim=1)  # F x B P C, P means seg_len
        res = torch.stack(res, dim=1).reshape(B, self.freq, -1)  # B F PC

        res = self.encoder(res)  # B F H
        x_rec, x_pred = self.dynamics(res, self.step)  # B F H, B S H

        x_rec = self.decoder(x_rec)  # B F PC
        x_rec = x_rec.reshape(B, self.freq, self.seg_len, self.enc_in)
        x_rec = x_rec.reshape(B, -1, self.enc_in)[:, : self.input_len, :]  # B L C

        x_pred = self.decoder(x_pred)  # B S PC
        x_pred = x_pred.reshape(B, self.step, self.seg_len, self.enc_in)
        x_pred = x_pred.reshape(B, -1, self.enc_in)[:, : self.pred_len, :]  # B S C

        return x_rec, x_pred


class TimeInvKP(nn.Module):
    """Koopman predictor with a learnable Koopman operator.

    Uses lookback and forecast window snapshots to predict the future of the
    time-invariant term.
    """

    def __init__(
        self,
        input_len=96,
        pred_len=96,
        dynamic_dim=128,
        encoder=None,
        decoder=None,
    ):
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.input_len = input_len
        self.pred_len = pred_len
        self.encoder = encoder
        self.decoder = decoder

        K_init = torch.randn(self.dynamic_dim, self.dynamic_dim)
        U, _, V = torch.svd(K_init)  # stable initialization
        self.K = nn.Linear(self.dynamic_dim, self.dynamic_dim, bias=False)
        self.K.weight.data = torch.mm(U, V.t())

    def forward(self, x):
        # x: B L C
        res = x.transpose(1, 2)  # B C L
        res = self.encoder(res)  # B C H
        res = self.K(res)  # B C H
        res = self.decoder(res)  # B C S
        res = res.transpose(1, 2)  # B S C
        return res


class Model(nn.Module):
    """Koopa forecaster.

    Paper: https://arxiv.org/pdf/2305.18803.pdf
    """

    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        label_len=0,
        features="M",
        seg_len=None,
        dynamic_dim=128,
        hidden_dim=64,
        hidden_layers=2,
        num_blocks=3,
        multistep=False,
        alpha=0.2,
    ):
        super().__init__()
        self.enc_in = enc_in
        self.input_len = seq_len
        self.pred_len = pred_len
        self.features = features

        # Upstream sets seg_len = pred_len; keep that as the default.
        self.seg_len = pred_len if seg_len is None else seg_len
        self.num_blocks = num_blocks
        self.dynamic_dim = dynamic_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.multistep = multistep
        self.alpha = alpha

        # mask_spectrum is estimated lazily from the first forward batch since
        # the model factory has no access to the training DataLoader.
        self.mask_spectrum = None
        self.disentanglement = None

        # shared encoder/decoder to make koopman embedding consistent
        self.time_inv_encoder = MLP(
            f_in=self.input_len,
            f_out=self.dynamic_dim,
            activation="relu",
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
        )
        self.time_inv_decoder = MLP(
            f_in=self.dynamic_dim,
            f_out=self.pred_len,
            activation="relu",
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
        )
        self.time_inv_kps = nn.ModuleList(
            [
                TimeInvKP(
                    input_len=self.input_len,
                    pred_len=self.pred_len,
                    dynamic_dim=self.dynamic_dim,
                    encoder=self.time_inv_encoder,
                    decoder=self.time_inv_decoder,
                )
                for _ in range(self.num_blocks)
            ]
        )

        # shared encoder/decoder to make koopman embedding consistent
        self.time_var_encoder = MLP(
            f_in=self.seg_len * self.enc_in,
            f_out=self.dynamic_dim,
            activation="tanh",
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
        )
        self.time_var_decoder = MLP(
            f_in=self.dynamic_dim,
            f_out=self.seg_len * self.enc_in,
            activation="tanh",
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
        )
        self.time_var_kps = nn.ModuleList(
            [
                TimeVarKP(
                    enc_in=self.enc_in,
                    input_len=self.input_len,
                    pred_len=self.pred_len,
                    seg_len=self.seg_len,
                    dynamic_dim=self.dynamic_dim,
                    encoder=self.time_var_encoder,
                    decoder=self.time_var_decoder,
                    multistep=self.multistep,
                )
                for _ in range(self.num_blocks)
            ]
        )

    def _init_mask_spectrum(self, x_enc):
        """Estimate the shared low-frequency mask from a batch of lookbacks."""
        amps = abs(torch.fft.rfft(x_enc, dim=1)).mean(dim=0).mean(dim=1)
        k = max(1, int(amps.shape[0] * self.alpha))
        self.mask_spectrum = amps.topk(k).indices
        self.disentanglement = FourierFilter(self.mask_spectrum)

    def forecast(self, x_enc):
        # Series Stationarization adopted from NSformer
        mean_enc = x_enc.mean(1, keepdim=True).detach()  # B x 1 x E
        x_enc = x_enc - mean_enc
        std_enc = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x_enc = x_enc / std_enc

        # Koopman Forecasting
        residual, forecast = x_enc, None
        for i in range(self.num_blocks):
            time_var_input, time_inv_input = self.disentanglement(residual)
            time_inv_output = self.time_inv_kps[i](time_inv_input)
            time_var_backcast, time_var_output = self.time_var_kps[i](
                time_var_input
            )
            residual = residual - time_var_backcast
            if forecast is None:
                forecast = time_inv_output + time_var_output
            else:
                forecast += time_inv_output + time_var_output

        # Series Stationarization adopted from NSformer
        res = forecast * std_enc + mean_enc
        return res

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.mask_spectrum is None:
            self._init_mask_spectrum(x_enc)
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len :, :]  # [B, L, D]
