"""HDMixer model implementation.

Vendored/adapted from https://github.com/hqh0728/HDMixer
(models/HDMixer.py, layers/box_coder1D.py, layers/PatchTST_layers.py).

HDMixer: Hierarchical Dependency with Extendable Patch for Multivariate Time
Series Forecasting (AAAI 2024).

License note: the upstream repository ships NO top-level LICENSE file (all
rights reserved by the authors); ``layers/box_coder1D.py`` carries a permissive
Facebook/Meta copyright header. The architecture is reimplemented here for the
ModernTSF (B, T, C) contract.

Adapted for ModernTSF:
- the upstream ``configs``-object constructor is replaced with plain keyword
  arguments;
- the shared ``RevIN`` layer under ``models.module.revin`` is reused;
- the small ``Transpose`` / ``get_activation_fn`` / ``positional_encoding``
  helpers and the ``pointwhCoder`` Length-Extendable-Patcher are vendored
  locally (they are HDMixer-specific and not present in ``models.module``);
- the auxiliary patch-entropy REINFORCE loss (``PaEN_Loss``) used only during
  upstream training is dropped; ``forward`` returns the prediction tensor only;
- only the long-term-forecast path is kept.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.module.revin import RevIN


# --------------------------------------------------------------------------- #
# Small vendored helpers (from layers/PatchTST_layers.py)
# --------------------------------------------------------------------------- #
class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


def get_activation_fn(activation):
    if callable(activation):
        return activation()
    if activation.lower() == "relu":
        return nn.ReLU()
    if activation.lower() == "gelu":
        return nn.GELU()
    raise ValueError(f'{activation} is not available. Use "relu", "gelu", or a callable.')


def positional_encoding(pe, learn_pe, q_len, d_model):
    if pe is None:
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == "zero":
        W_pos = torch.empty((q_len, 1))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe == "zeros":
        W_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(W_pos, -0.02, 0.02)
    elif pe in ("normal", "gauss"):
        W_pos = torch.zeros((q_len, 1))
        torch.nn.init.normal_(W_pos, mean=0.0, std=0.1)
    elif pe == "uniform":
        W_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(W_pos, a=0.0, b=0.1)
    else:
        raise ValueError(f"{pe} is not a valid pe.")
    return nn.Parameter(W_pos, requires_grad=learn_pe)


# --------------------------------------------------------------------------- #
# Length-Extendable Patcher (from layers/box_coder1D.py)
# --------------------------------------------------------------------------- #
class pointCoder(nn.Module):
    def __init__(self, input_size, patch_count, weights=(1.0, 1.0, 1.0), tanh=True):
        super().__init__()
        self.input_size = input_size
        self.patch_count = patch_count
        self.weights = weights
        self.tanh = tanh

    def _generate_anchor(self, device="cpu"):
        anchors = []
        patch_stride_x = 2.0 / self.patch_count
        for i in range(self.patch_count):
            x = -1 + (0.5 + i) * patch_stride_x
            anchors.append([x])
        self.anchor = torch.as_tensor(anchors, device=device)


class pointwhCoder(pointCoder):
    def __init__(
        self,
        input_size,
        patch_count,
        weights=(1.0, 1.0, 1.0),
        pts=1,
        tanh=True,
        wh_bias=None,
        deform_range=0.25,
    ):
        super().__init__(
            input_size=input_size, patch_count=patch_count, weights=weights, tanh=tanh
        )
        self.patch_pixel = pts
        self.wh_bias = None
        if wh_bias is not None:
            self.wh_bias = nn.Parameter(torch.zeros(2) + wh_bias)
        self.deform_range = deform_range

    def forward(self, boxes):
        self._generate_anchor(device=boxes.device)
        if self.wh_bias is not None:
            boxes[:, :, 1:] = boxes[:, :, 1:] + self.wh_bias
        self.boxes = self.decode(boxes)
        points = self.meshgrid(self.boxes)
        return points

    def decode(self, rel_codes):
        boxes = self.anchor
        pixel_x = 2.0 / self.patch_count
        wx, ww1, ww2 = self.weights

        dx = (
            torch.tanh(rel_codes[:, :, 0] / wx) * pixel_x / 4
            if self.tanh
            else rel_codes[:, :, 0] * pixel_x / wx
        )
        dw1 = F.relu(torch.tanh(rel_codes[:, :, 1] / ww1)) * pixel_x * self.deform_range + pixel_x
        dw2 = F.relu(torch.tanh(rel_codes[:, :, 2] / ww2)) * pixel_x * self.deform_range + pixel_x

        pred_boxes = torch.zeros(
            (rel_codes.shape[0], rel_codes.shape[1], rel_codes.shape[2] - 1)
        ).to(rel_codes.device)
        ref_x = boxes[:, 0].unsqueeze(0)
        pred_boxes[:, :, 0] = dx + ref_x - dw1
        pred_boxes[:, :, 1] = dx + ref_x + dw2
        pred_boxes = pred_boxes.clamp_(min=-1.0, max=1.0)
        return pred_boxes

    def meshgrid(self, boxes):
        B = boxes.shape[0]
        xs = boxes
        xs = torch.nn.functional.interpolate(
            xs, size=self.patch_pixel, mode="linear", align_corners=True
        )
        results = xs.reshape(B, self.patch_count, self.patch_pixel, 1)
        return results


# --------------------------------------------------------------------------- #
# Hierarchical Dependency Explorer (HDE)
# --------------------------------------------------------------------------- #
class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        B, M, D, N = x.shape
        x = x.reshape(B * M, D, N)
        x = self.norm(x)
        x = x.reshape(B, M, D, N)
        return x


class HDMixerLayer(nn.Module):
    def __init__(
        self,
        c_in,
        q_len,
        d_model,
        d_ff=256,
        dropout=0.0,
        bias=True,
        activation="gelu",
        mix_time=True,
        mix_variable=True,
        mix_channel=True,
    ):
        super().__init__()
        self.mix_time = mix_time
        self.mix_variable = mix_variable
        self.mix_channel = mix_channel
        self.patch_mixer = nn.Sequential(
            LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model, bias=bias),
            nn.Dropout(dropout),
        )
        self.time_mixer = nn.Sequential(
            Transpose(2, 3),
            LayerNorm(q_len),
            nn.Linear(q_len, q_len * 2, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(q_len * 2, q_len, bias=bias),
            nn.Dropout(dropout),
            Transpose(2, 3),
        )
        self.variable_mixer = nn.Sequential(
            Transpose(1, 3),
            LayerNorm(c_in),
            nn.Linear(c_in, c_in * 2, bias=bias),
            get_activation_fn(activation),
            nn.Dropout(dropout),
            nn.Linear(c_in * 2, c_in, bias=bias),
            nn.Dropout(dropout),
            Transpose(1, 3),
        )

    def forward(self, src):
        # src: [bs x nvars x patch_num x d_model]
        u = self.patch_mixer(src) + src if self.mix_channel else src
        v = self.time_mixer(u) + src if self.mix_time else u
        w = self.variable_mixer(v) + src if self.mix_variable else v
        return w


class HDMixerStack(nn.Module):
    def __init__(
        self,
        c_in,
        q_len,
        d_model,
        d_ff=None,
        dropout=0.0,
        activation="gelu",
        n_layers=1,
        mix_time=True,
        mix_variable=True,
        mix_channel=True,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                HDMixerLayer(
                    c_in,
                    q_len,
                    d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    activation=activation,
                    mix_time=mix_time,
                    mix_variable=mix_variable,
                    mix_channel=mix_channel,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, src):
        output = src
        for mod in self.layers:
            output = mod(output)
        return output


class Encoder(nn.Module):
    def __init__(
        self,
        c_in,
        patch_num,
        patch_len,
        n_layers=3,
        d_model=128,
        d_ff=256,
        dropout=0.0,
        act="gelu",
        pe="zeros",
        learn_pe=True,
        mix_time=True,
        mix_variable=True,
        mix_channel=True,
    ):
        super().__init__()
        self.patch_num = patch_num
        self.patch_len = patch_len
        q_len = patch_num
        self.W_P = nn.Linear(patch_len, d_model)
        self.W_pos = positional_encoding(pe, learn_pe, q_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.encoder = HDMixerStack(
            c_in,
            q_len,
            d_model,
            d_ff=d_ff,
            dropout=dropout,
            activation=act,
            n_layers=n_layers,
            mix_time=mix_time,
            mix_variable=mix_variable,
            mix_channel=mix_channel,
        )

    def forward(self, x):
        # x: [bs x nvars x patch_len x patch_num]
        x = x.permute(0, 1, 3, 2)  # [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)  # [bs x nvars x patch_num x d_model]
        z = self.encoder(x)  # [bs x nvars x patch_num x d_model]
        return z


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.individual = individual
        self.n_vars = n_vars
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for _ in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x patch_num x d_model]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])
                z = self.linears[i](z)
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x


class HDMixer_backbone(nn.Module):
    def __init__(
        self,
        c_in,
        context_window,
        target_window,
        patch_len,
        stride,
        n_layers=3,
        d_model=128,
        d_ff=256,
        dropout=0.0,
        act="gelu",
        head_dropout=0.0,
        individual=False,
        revin=True,
        affine=True,
        subtract_last=False,
        deform_range=0.25,
        pe="zeros",
        learn_pe=True,
        mix_time=True,
        mix_variable=True,
        mix_channel=True,
    ):
        super().__init__()
        self.n_vars = c_in
        self.revin = revin
        if self.revin:
            self.revin_layer = RevIN(c_in, affine=affine, subtract_last=subtract_last)

        self.patch_len = patch_len
        self.stride = stride
        # Length-Extendable Patcher: patch_num = context_window // stride
        self.patch_num = patch_num = context_window // stride
        self.patch_shift_linear = nn.Linear(context_window, self.patch_num * 3)
        self.box_coder = pointwhCoder(
            input_size=context_window,
            patch_count=self.patch_num,
            weights=(1.0, 1.0, 1.0),
            pts=self.patch_len,
            tanh=True,
            wh_bias=torch.tensor(5.0 / 3.0).sqrt().log(),
            deform_range=deform_range,
        )

        self.backbone = Encoder(
            c_in,
            patch_num=patch_num,
            patch_len=patch_len,
            n_layers=n_layers,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            act=act,
            pe=pe,
            learn_pe=learn_pe,
            mix_time=mix_time,
            mix_variable=mix_variable,
            mix_channel=mix_channel,
        )

        self.head_nf = d_model * patch_num
        self.head = Flatten_Head(
            individual, self.n_vars, self.head_nf, target_window, head_dropout=head_dropout
        )

    def forward(self, z):  # z: [bs x nvars x seq_len]
        batch_size = z.shape[0]
        seq_len = z.shape[-1]
        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, "norm")
            z = z.permute(0, 2, 1)

        # Length-Extendable Patcher (deformable patch sampling)
        anchor_shift = self.patch_shift_linear(z).view(
            batch_size * self.n_vars, self.patch_num, 3
        )
        sampling_location_1d = self.box_coder(anchor_shift)
        add1d = torch.ones(
            size=(batch_size * self.n_vars, self.patch_num, self.patch_len, 1),
            device=sampling_location_1d.device,
        ).float()
        sampling_location_2d = torch.cat([sampling_location_1d, add1d], dim=-1)
        z_grid = z.reshape(batch_size * self.n_vars, 1, 1, seq_len)
        patch = F.grid_sample(
            z_grid,
            sampling_location_2d,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).squeeze(1)  # B*C, patch_num, patch_len
        x_lep = patch.reshape(
            batch_size, self.n_vars, self.patch_num, self.patch_len
        ).permute(0, 1, 3, 2)  # [bs x nvars x patch_len x patch_num]

        z = self.backbone(x_lep)  # [bs x nvars x patch_num x d_model]
        z = self.head(z)  # [bs x nvars x target_window]

        if self.revin:
            z = z.permute(0, 2, 1)
            z = self.revin_layer(z, "denorm")
            z = z.permute(0, 2, 1)
        return z


class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        features="M",
        d_model=128,
        d_ff=256,
        e_layers=3,
        patch_len=16,
        stride=8,
        dropout=0.1,
        head_dropout=0.0,
        activation="gelu",
        individual=False,
        revin=True,
        affine=True,
        subtract_last=False,
        deform_range=0.25,
        mix_time=True,
        mix_variable=True,
        mix_channel=True,
    ):
        super().__init__()
        self.features = features
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.model = HDMixer_backbone(
            c_in=enc_in,
            context_window=seq_len,
            target_window=pred_len,
            patch_len=patch_len,
            stride=stride,
            n_layers=e_layers,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout,
            act=activation,
            head_dropout=head_dropout,
            individual=individual,
            revin=revin,
            affine=affine,
            subtract_last=subtract_last,
            deform_range=deform_range,
            mix_time=mix_time,
            mix_variable=mix_variable,
            mix_channel=mix_channel,
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x_enc: [B, seq_len, C]
        x = x_enc.permute(0, 2, 1)  # [B, C, seq_len]
        x = self.model(x)  # [B, C, pred_len]
        x = x.permute(0, 2, 1)  # [B, pred_len, C]
        return x[:, -self.pred_len :, :]
