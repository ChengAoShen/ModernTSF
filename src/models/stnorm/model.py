"""ModernTSF adapter for the STNorm spatiotemporal forecasting model.

Vendored/adapted from https://github.com/GestaltCogTeam/BasicTS
(baselines/STNorm), Apache-2.0.

STNorm (ST-Norm: Spatial and Temporal Normalization for Multi-variate Time
Series Forecasting, KDD 2021) applies spatial and temporal normalization on a
WaveNet-style dilated-convolution backbone. It requires **no** external
adjacency matrix.

The upstream architecture expects a history tensor of shape ``(B, L, N, C)``
where channel 0 is the value and the remaining channels are calendar
covariates. Its forward signature is
``forward(history_data, future_data, batch_seen, epoch, train, **kwargs)`` and
it returns ``(B, pred_len, N, 1)``.

This adapter converts ModernTSF's ``(x_enc, x_mark_enc)`` into that layout via
``to_spatiotemporal`` and squeezes the output channel back to
``(B, pred_len, N)``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models._external.marks import to_spatiotemporal


# ---------------------------------------------------------------------------
# Vendored upstream architecture (BasicTS baselines/STNorm/arch/stnorm_arch.py).
# Adjusted only to drop the unused module list and keep tensors on the input
# device (the original had no hardcoded cuda, so no device edits were needed).
# ---------------------------------------------------------------------------


class SNorm(nn.Module):
    """Spatial normalization over the node axis."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(channels))
        self.gamma = nn.Parameter(torch.ones(channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = (x - x.mean(2, keepdims=True)) / (
            x.var(2, keepdims=True, unbiased=True) + 0.00001
        ) ** 0.5
        out = x_norm * self.gamma.view(1, -1, 1, 1) + self.beta.view(1, -1, 1, 1)
        return out


class TNorm(nn.Module):
    """Temporal normalization with running statistics."""

    def __init__(
        self,
        num_nodes: int,
        channels: int,
        track_running_stats: bool = True,
        momentum: float = 0.1,
    ) -> None:
        super().__init__()
        self.track_running_stats = track_running_stats
        self.beta = nn.Parameter(torch.zeros(1, channels, num_nodes, 1))
        self.gamma = nn.Parameter(torch.ones(1, channels, num_nodes, 1))
        self.register_buffer("running_mean", torch.zeros(1, channels, num_nodes, 1))
        self.register_buffer("running_var", torch.ones(1, channels, num_nodes, 1))
        self.momentum = momentum

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.track_running_stats:
            mean = x.mean((0, 3), keepdims=True)
            var = x.var((0, 3), keepdims=True, unbiased=False)
            if self.training:
                n = x.shape[3] * x.shape[0]
                with torch.no_grad():
                    self.running_mean = (
                        self.momentum * mean + (1 - self.momentum) * self.running_mean
                    )
                    self.running_var = (
                        self.momentum * var * n / (n - 1)
                        + (1 - self.momentum) * self.running_var
                    )
            else:
                mean = self.running_mean
                var = self.running_var
        else:
            mean = x.mean((3), keepdims=True)
            var = x.var((3), keepdims=True, unbiased=True)
        x_norm = (x - mean) / (var + 0.00001) ** 0.5
        out = x_norm * self.gamma + self.beta
        return out


class STNorm(nn.Module):
    """ST-Norm WaveNet backbone.

    Paper: ST-Norm: Spatial and Temporal Normalization for Multi-variate Time
    Series Forecasting (SIGKDD 2021).
    """

    def __init__(
        self,
        num_nodes: int,
        tnorm_bool: bool,
        snorm_bool: bool,
        in_dim: int,
        out_dim: int,
        channels: int,
        kernel_size: int,
        blocks: int,
        layers: int,
    ) -> None:
        super().__init__()
        self.blocks = blocks
        self.layers = layers
        self.snorm_bool = snorm_bool
        self.tnorm_bool = tnorm_bool

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        if self.snorm_bool:
            self.sn = nn.ModuleList()
        if self.tnorm_bool:
            self.tn = nn.ModuleList()
        num = int(self.tnorm_bool) + int(self.snorm_bool) + 1

        self.start_conv = nn.Conv2d(
            in_channels=in_dim, out_channels=channels, kernel_size=(1, 1)
        )

        receptive_field = 1
        self.dropout = nn.Dropout(0.2)
        self.dilation = []

        for _b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for _i in range(layers):
                self.dilation.append(new_dilation)
                if self.tnorm_bool:
                    self.tn.append(TNorm(num_nodes, channels))
                if self.snorm_bool:
                    self.sn.append(SNorm(channels))
                self.filter_convs.append(
                    nn.Conv2d(
                        in_channels=num * channels,
                        out_channels=channels,
                        kernel_size=(1, kernel_size),
                        dilation=new_dilation,
                    )
                )
                self.gate_convs.append(
                    nn.Conv2d(
                        in_channels=num * channels,
                        out_channels=channels,
                        kernel_size=(1, kernel_size),
                        dilation=new_dilation,
                    )
                )
                self.residual_convs.append(
                    nn.Conv2d(
                        in_channels=channels, out_channels=channels, kernel_size=(1, 1)
                    )
                )
                self.skip_convs.append(
                    nn.Conv2d(
                        in_channels=channels, out_channels=channels, kernel_size=(1, 1)
                    )
                )
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2

        self.end_conv_1 = nn.Conv2d(
            in_channels=channels, out_channels=channels, kernel_size=(1, 1), bias=True
        )
        self.end_conv_2 = nn.Conv2d(
            in_channels=channels, out_channels=out_dim, kernel_size=(1, 1), bias=True
        )
        self.receptive_field = receptive_field

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor,
        batch_seen: int,
        epoch: int,
        train: bool,
        **kwargs,
    ) -> torch.Tensor:
        """Feedforward.

        Args:
            history_data: shape ``(B, L, N, C)``.

        Returns:
            torch.Tensor of shape ``(B, out_dim, N, 1)``.
        """
        input = history_data.transpose(1, 3).contiguous()  # (B, C, N, L)
        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = F.pad(input, (self.receptive_field - in_len, 0, 0, 0))
        else:
            x = input
        x = self.start_conv(x)
        skip = 0

        for i in range(self.blocks * self.layers):
            residual = x
            x_list = [x]
            if self.tnorm_bool:
                x_list.append(self.tn[i](x))
            if self.snorm_bool:
                x_list.append(self.sn[i](x))
            x = torch.cat(x_list, dim=1)
            filter = torch.tanh(self.filter_convs[i](x))
            gate = torch.sigmoid(self.gate_convs[i](x))
            x = filter * gate
            s = self.skip_convs[i](x)
            try:
                skip = skip[:, :, :, -s.size(3):]
            except (TypeError, AttributeError):
                skip = 0
            skip = s + skip
            x = self.residual_convs[i](x)
            x = x + residual[:, :, :, -x.size(3):]

        x = F.relu(skip)
        rep = F.relu(self.end_conv_1(x))
        out = self.end_conv_2(rep)
        return out


# ---------------------------------------------------------------------------
# ModernTSF adapter.
# ---------------------------------------------------------------------------


class Model(nn.Module):
    """Adapter wrapping the upstream STNorm backbone.

    Parameters
    ----------
    seq_len : int
        Input sequence length.
    pred_len : int
        Forecast horizon (= upstream ``out_dim``).
    num_nodes : int
        Number of spatial nodes.
    adj_mx : np.ndarray, optional
        Adjacency matrix. STNorm needs no graph, so this is accepted and
        ignored for interface compatibility with the Stage-4 pipeline.
    input_dim : int
        Number of input channels fed to the backbone (value + calendar feats).
    channels : int
        Hidden channel width.
    kernel_size : int
        Dilated-convolution temporal kernel size.
    blocks : int
        Number of WaveNet blocks.
    layers : int
        Number of dilated layers per block.
    tnorm_bool : bool
        Enable temporal normalization.
    snorm_bool : bool
        Enable spatial normalization.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx=None,
        input_dim: int = 3,
        channels: int = 16,
        kernel_size: int = 2,
        blocks: int = 2,
        layers: int = 2,
        tnorm_bool: bool = True,
        snorm_bool: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        # adj_mx is intentionally unused: STNorm is graph-free.
        self.net = STNorm(
            num_nodes=num_nodes,
            tnorm_bool=tnorm_bool,
            snorm_bool=snorm_bool,
            in_dim=input_dim,
            out_dim=pred_len,
            channels=channels,
            kernel_size=kernel_size,
            blocks=blocks,
            layers=layers,
        )

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
        x_mark_enc : torch.Tensor, optional
            Marks: raw ``(B, seq_len, 6)`` stamps or node-structured
            ``(B, seq_len, N, F)`` covariates.

        Returns
        -------
        torch.Tensor
            Forecast of shape ``(B, pred_len, N)``.
        """
        history = to_spatiotemporal(x_enc, x_mark_enc)  # (B, L, N, 1+F)
        # Match the backbone's expected input channel count.
        c = history.shape[-1]
        if c < self.input_dim:
            pad = history.new_zeros(
                (*history.shape[:-1], self.input_dim - c)
            )
            history = torch.cat([history, pad], dim=-1)
        elif c > self.input_dim:
            history = history[..., : self.input_dim]
        out = self.net(
            history,
            None,
            batch_seen=0,
            epoch=0,
            train=self.training,
        )  # (B, pred_len, N, 1)
        if out.dim() == 4:
            return out[..., 0]
        return out.reshape(out.shape[0], self.pred_len, self.num_nodes)
