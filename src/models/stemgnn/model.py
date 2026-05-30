"""ModernTSF adapter for the StemGNN spectral-temporal graph forecaster.

Vendored/adapted from https://github.com/GestaltCogTeam/BasicTS
(baselines/StemGNN), Apache-2.0.

StemGNN (NeurIPS 2020, "Spectral Temporal Graph Neural Network for
Multivariate Time-series Forecasting") learns a *latent* correlation graph
internally via self-attention over a GRU summary of the input — it does not
require an external adjacency matrix. The graph is turned into a normalized
Laplacian, expanded into Chebyshev polynomials, and combined with a Graph
Fourier Transform + DFT spectral block (the StockBlockLayer) to forecast.

This adapter converts ModernTSF's ``(x_enc, marks)`` into the BasicTS
spatiotemporal layout ``(B, L, N, 1 + F)`` (channel 0 = value), drives the
upstream module with the BasicTS forward signature, and squeezes the single
output channel back to ``(B, pred_len, N)``. The upstream arch only consumes
channel 0, so any calendar / meteorology covariates are ignored.

All tensors created internally use the input tensor's device (the upstream
arch is already device-safe), so no ``.cuda()`` calls remain.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models._external.marks import to_spatiotemporal


# --------------------------------------------------------------------------- #
# Vendored upstream arch (BasicTS baselines/StemGNN/arch/stemgnn_arch.py).
# Imports adjusted to be self-contained; no hardcoded device usage.
# --------------------------------------------------------------------------- #


class GLU(nn.Module):
    def __init__(self, input_channel, output_channel):
        super().__init__()
        self.linear_left = nn.Linear(input_channel, output_channel)
        self.linear_right = nn.Linear(input_channel, output_channel)

    def forward(self, x):
        return torch.mul(self.linear_left(x), torch.sigmoid(self.linear_right(x)))


class StockBlockLayer(nn.Module):
    def __init__(self, time_step, unit, multi_layer, stack_cnt=0):
        super().__init__()
        self.time_step = time_step
        self.unit = unit
        self.stack_cnt = stack_cnt
        self.multi = multi_layer
        self.weight = nn.Parameter(
            torch.Tensor(1, 3 + 1, 1, self.time_step * self.multi, self.multi * self.time_step)
        )  # [K+1, 1, in_c, out_c]
        nn.init.xavier_normal_(self.weight)
        self.forecast = nn.Linear(self.time_step * self.multi, self.time_step * self.multi)
        self.forecast_result = nn.Linear(self.time_step * self.multi, self.time_step)
        if self.stack_cnt == 0:
            self.backcast = nn.Linear(self.time_step * self.multi, self.time_step)
        self.backcast_short_cut = nn.Linear(self.time_step, self.time_step)
        self.relu = nn.ReLU()
        self.GLUs = nn.ModuleList()
        self.output_channel = 4 * self.multi
        for i in range(3):
            if i == 0:
                self.GLUs.append(GLU(self.time_step * 4, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * 4, self.time_step * self.output_channel))
            elif i == 1:
                self.GLUs.append(
                    GLU(self.time_step * self.output_channel, self.time_step * self.output_channel)
                )
                self.GLUs.append(
                    GLU(self.time_step * self.output_channel, self.time_step * self.output_channel)
                )
            else:
                self.GLUs.append(
                    GLU(self.time_step * self.output_channel, self.time_step * self.output_channel)
                )
                self.GLUs.append(
                    GLU(self.time_step * self.output_channel, self.time_step * self.output_channel)
                )

    def spe_seq_cell(self, input):
        batch_size, k, input_channel, node_cnt, time_step = input.size()
        input = input.view(batch_size, -1, node_cnt, time_step)
        ffted = torch.fft.fft(input, dim=-1)
        ffted_real = ffted.real
        ffted_imag = ffted.imag
        ffted = torch.stack([ffted_real, ffted_imag], dim=-1)
        real = ffted[..., 0].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)
        img = ffted[..., 1].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)
        for i in range(3):
            real = self.GLUs[i * 2](real)
            img = self.GLUs[2 * i + 1](img)
        real = real.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()
        img = img.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()
        time_step_as_inner = torch.complex(real, img)
        iffted = torch.fft.ifft(time_step_as_inner, dim=-1).real
        return iffted

    def forward(self, x, mul_L):
        mul_L = mul_L.unsqueeze(1)
        x = x.unsqueeze(1)
        gfted = torch.matmul(mul_L, x)  # B, cheb_order, 1, N, L
        gconv_input = self.spe_seq_cell(gfted).unsqueeze(2)
        igfted = torch.matmul(gconv_input, self.weight)
        igfted = torch.sum(igfted, dim=1)
        forecast_source = torch.sigmoid(self.forecast(igfted).squeeze(1))
        forecast = self.forecast_result(forecast_source)
        if self.stack_cnt == 0:
            backcast_short = self.backcast_short_cut(x).squeeze(1)
            backcast_source = torch.sigmoid(self.backcast(igfted) - backcast_short)
        else:
            backcast_source = None
        return forecast, backcast_source


class StemGNN(nn.Module):
    """Spectral-temporal graph neural network (BasicTS implementation)."""

    def __init__(
        self,
        units,
        stack_cnt,
        time_step,
        multi_layer,
        horizon,
        dropout_rate=0.5,
        leaky_rate=0.2,
        **kwargs,
    ):
        super().__init__()
        self.unit = units
        self.stack_cnt = stack_cnt
        self.alpha = leaky_rate
        self.time_step = time_step
        self.horizon = horizon
        self.weight_key = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(self.unit, 1)))
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)
        self.GRU = nn.GRU(self.time_step, self.unit)
        self.multi_layer = multi_layer
        self.stock_block = nn.ModuleList()
        self.stock_block.extend(
            [
                StockBlockLayer(self.time_step, self.unit, self.multi_layer, stack_cnt=i)
                for i in range(self.stack_cnt)
            ]
        )
        self.fc = nn.Sequential(
            nn.Linear(int(self.time_step), int(self.time_step)),
            nn.LeakyReLU(),
            nn.Linear(int(self.time_step), self.horizon),
        )
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(p=dropout_rate)

    def get_laplacian(self, graph, normalize):
        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(
                graph.size(0), device=graph.device, dtype=graph.dtype
            ) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L

    def cheb_polynomial(self, laplacian):
        N = laplacian.size(0)  # [N, N]
        laplacian = laplacian.unsqueeze(0)
        first_laplacian = torch.zeros(
            [1, N, N], device=laplacian.device, dtype=torch.float
        )
        second_laplacian = laplacian
        third_laplacian = (2 * torch.matmul(laplacian, second_laplacian)) - first_laplacian
        forth_laplacian = 2 * torch.matmul(laplacian, third_laplacian) - second_laplacian
        multi_order_laplacian = torch.cat(
            [first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=0
        )
        return multi_order_laplacian

    def latent_correlation_layer(self, x):
        input, _ = self.GRU(x.permute(2, 0, 1).contiguous())
        input = input.permute(1, 0, 2).contiguous()
        attention = self.self_graph_attention(input)
        attention = torch.mean(attention, dim=0)
        degree = torch.sum(attention, dim=1)
        attention = 0.5 * (attention + attention.T)
        degree_l = torch.diag(degree)
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-7))
        laplacian = torch.matmul(
            diagonal_degree_hat, torch.matmul(degree_l - attention, diagonal_degree_hat)
        )
        mul_L = self.cheb_polynomial(laplacian)
        return mul_L, attention

    def self_graph_attention(self, input):
        input = input.permute(0, 2, 1).contiguous()
        bat, N, fea = input.size()
        key = torch.matmul(input, self.weight_key)
        query = torch.matmul(input, self.weight_query)
        data = key.repeat(1, 1, N).view(bat, N * N, 1) + query.repeat(1, N, 1)
        data = data.squeeze(2)
        data = data.view(bat, N, -1)
        data = self.leakyrelu(data)
        attention = F.softmax(data, dim=2)
        attention = self.dropout(attention)
        return attention

    def graph_fft(self, input, eigenvectors):
        return torch.matmul(eigenvectors, input)

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor,
        batch_seen: int,
        epoch: int,
        train: bool,
        **kwargs,
    ) -> torch.Tensor:
        """Feedforward function of StemGNN.

        Args:
            history_data (torch.Tensor): [B, L, N, C]; channel 0 is the value.

        Returns:
            torch.Tensor: [B, horizon, N, 1]
        """
        x = history_data[..., 0]
        mul_L, attention = self.latent_correlation_layer(x)
        X = x.unsqueeze(1).permute(0, 1, 3, 2).contiguous()
        result = []
        for stack_i in range(self.stack_cnt):
            forecast, X = self.stock_block[stack_i](X, mul_L)
            result.append(forecast)
        forecast = result[0] + result[1]
        forecast = self.fc(forecast)
        return forecast.permute(0, 2, 1).contiguous().unsqueeze(-1)


# --------------------------------------------------------------------------- #
# ModernTSF adapter
# --------------------------------------------------------------------------- #


class Model(nn.Module):
    """Adapter wrapping the upstream StemGNN model.

    Parameters
    ----------
    seq_len : int
        Input sequence length (becomes the upstream ``time_step``).
    pred_len : int
        Forecast horizon.
    num_nodes : int
        Number of spatial nodes ``N`` (becomes the upstream ``units``).
    adj_mx : np.ndarray, optional
        Predefined ``(N, N)`` adjacency. StemGNN learns its own latent graph
        via self-attention, so this is accepted for interface uniformity but
        not required / used.
    input_dim : int
        Number of input channels per node (only channel 0 is consumed).
    multi_layer : int
        Spectral block multiplier (controls hidden width).
    dropout_rate : float
        Dropout applied inside the attention graph builder.
    leaky_rate : float
        Negative slope for the LeakyReLU attention activation.
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx=None,
        input_dim: int = 3,
        multi_layer: int = 3,
        dropout_rate: float = 0.5,
        leaky_rate: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        # StemGNN's spectral combine hardcodes result[0] + result[1] -> 2 stacks.
        self.net = StemGNN(
            units=num_nodes,
            stack_cnt=2,
            time_step=seq_len,
            multi_layer=multi_layer,
            horizon=pred_len,
            dropout_rate=dropout_rate,
            leaky_rate=leaky_rate,
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
            Raw input marks ``(B, seq_len, 6)`` or node covariates
            ``(B, seq_len, N, F)``. Only channel 0 (the value) is consumed.

        Returns
        -------
        torch.Tensor
            Forecast of shape ``(B, pred_len, N)``.
        """
        history = to_spatiotemporal(x_enc, x_mark_enc)  # (B, L, N, 1+F)
        out = self.net(
            history, None, batch_seen=0, epoch=0, train=self.training
        )  # (B, pred_len, N, 1)
        if out.dim() == 4:
            return out[..., 0]
        return out.reshape(out.shape[0], self.pred_len, self.num_nodes)
