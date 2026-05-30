"""Sumba model implementation.

Vendored/adapted from https://github.com/chenxiaodanhit/Sumba
(models/Sumba.py, layers/TCN.py, layers/DynamicGCN.py). The upstream repository
ships no LICENSE file (license: null on GitHub, i.e. all rights reserved). Code
is attributed to the authors and adapted here for benchmarking only.

Sumba: "Structured Matrix Basis for Multivariate Time Series Forecasting with
Interpretable Dynamics" (NeurIPS 2024), Chen, Li, Chen, Li.

Adapted for ModernTSF:
- the upstream ``config``-object constructor is replaced with plain keyword
  arguments (``configs.X`` -> kwargs);
- the Multi-Scale TCN and Dynamic GCN blocks (upstream ``layers/TCN.py`` and
  ``layers/DynamicGCN.py``) are vendored locally since they are Sumba-specific
  and not present in ``models.module.*``;
- the hardcoded ``'cuda:0'`` device buffer is removed (device-agnostic);
- the temporal-mark channel count is parameterised (``mark_dim``) instead of the
  upstream hardcoded ``4`` (ModernTSF emits 6 calendar features);
- only the long-term multivariate forecast path is kept.

Operates on standard (B, T, C) multivariate LTSF tensors; ``num_nodes`` == enc_in.
"""

from __future__ import annotations

import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


# ---------------------------------------------------------------------------
# Vendored from layers/TCN.py
# ---------------------------------------------------------------------------
class Dilated_Inception(nn.Module):
    def __init__(self, cin, cout, kernel_set, dilation_factor, mark_dim):
        super().__init__()
        self.tconv = nn.ModuleList()
        self.timeconv = nn.ModuleList()
        self.kernel_set = kernel_set
        self.mark_dim = mark_dim
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))
        for kern in self.kernel_set:
            self.timeconv.append(
                nn.Conv1d(mark_dim, mark_dim, kern, dilation=dilation_factor)
            )
        self.timepro = nn.Conv1d(mark_dim * len(self.kernel_set), mark_dim, 1)

    def forward(self, input, x_mark_enc):
        x = []
        x_mark_enc = x_mark_enc.transpose(-1, -2)
        x_mark_enc_list = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
            x_mark_enc_list.append(self.timeconv[i](x_mark_enc))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3):]
            x_mark_enc_list[i] = x_mark_enc_list[i][..., -x_mark_enc_list[-1].size(2):]
        x = torch.cat(x, dim=1)
        x_mark_enc = torch.cat(x_mark_enc_list, dim=1)
        x_mark_enc = self.timepro(x_mark_enc)
        return x, x_mark_enc.transpose(-2, -1)


class LayerNorm(nn.Module):
    __constants__ = ["normalized_shape", "weight", "bias", "eps", "elementwise_affine"]

    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.Tensor(*normalized_shape))
            self.bias = nn.Parameter(torch.Tensor(*normalized_shape))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.elementwise_affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input):
        return F.layer_norm(input, tuple(input.shape[1:]), self.weight, self.bias, self.eps)


class TConv(nn.Module):
    def __init__(self, residual_channels, conv_channels, kernel_set, dilation_factor, dropout, mark_dim):
        super().__init__()
        self.filter_conv = Dilated_Inception(residual_channels, conv_channels, kernel_set, dilation_factor, mark_dim)
        self.gate_conv = Dilated_Inception(residual_channels, conv_channels, kernel_set, dilation_factor, mark_dim)
        self.dropout = dropout

    def forward(self, x, x_mark_enc):
        _filter, x_mark_enc_red = self.filter_conv(x, x_mark_enc)
        filt = torch.tanh(_filter)
        _gate, x_mark_enc_red = self.gate_conv(x, x_mark_enc)
        gate = torch.sigmoid(_gate)
        x = filt * gate
        x = F.dropout(x, self.dropout, training=self.training)
        return x, x_mark_enc_red


# ---------------------------------------------------------------------------
# Vendored from layers/DynamicGCN.py
# ---------------------------------------------------------------------------
class linear(nn.Module):
    def __init__(self, c_in, c_out, bias=True):
        super().__init__()
        self.mlp = nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0, 0), stride=(1, 1), bias=bias)

    def forward(self, x):
        return self.mlp(x)


class dynamicGCN(nn.Module):
    def __init__(self, c_in, c_out, gdep, dropout, alpha, K, num_nodes, dimension):
        super().__init__()
        self.mlp = linear(2 * c_in, c_out)
        self.gdep = gdep
        self.dropout = dropout
        self.alpha = alpha
        self.K = K

    def forward(self, x, U, V, weight, sigma):
        sigma_diag = torch.stack([torch.diag(var) for var in sigma], dim=0)
        inverse_coordinate = torch.einsum("kn,BTnd->BTkd", V.transpose(1, 0), x.transpose(3, 1))
        multiplication_matrix = torch.einsum("Mck,BTkd->BTMcd", sigma_diag, inverse_coordinate)
        dynamic_multiplication = torch.einsum("BTM,BTMcd->BTcd", weight, multiplication_matrix)
        origin_coordinate = torch.einsum("nc,BTcd->BTnd", U, dynamic_multiplication).transpose(3, 1)
        out = [x]
        h = self.alpha * x + (1 - self.alpha) * origin_coordinate
        out.append(h)
        ho = torch.cat(out, dim=1)
        ho = self.mlp(ho)
        return ho


# ---------------------------------------------------------------------------
# Vendored from models/Sumba.py
# ---------------------------------------------------------------------------
class Extractor(nn.Module):
    def __init__(self, residual_channels, conv_channels, kernel_set, dilation_factor, gcn_depth,
                 M, dy_embedding_dim, skip_channels, t_len, num_nodes, layer_norm_affline,
                 propalpha, dropout, D, LowRank, mark_dim):
        super().__init__()
        self.t_conv = TConv(residual_channels, conv_channels, kernel_set, dilation_factor, dropout, mark_dim)
        self.skip_conv = nn.Conv2d(conv_channels, skip_channels, kernel_size=(1, t_len))
        self.s_conv = dynamicGCN(conv_channels, residual_channels, gcn_depth, dropout, propalpha, M, num_nodes, dy_embedding_dim)
        self.residual_conv = nn.Conv2d(conv_channels, residual_channels, kernel_size=(1, 1))
        self.norm = LayerNorm((residual_channels, num_nodes, t_len), elementwise_affine=layer_norm_affline)
        self.D = D
        self.Linear_query = nn.Linear(dy_embedding_dim * num_nodes + mark_dim, D)
        self.Linear_key = nn.Linear(LowRank, D)

    def weight_generation(self, x, sigma, x_mark_enc):
        B, C, N, T = x.shape
        x = x.transpose(1, 3)  # BTNC
        x = x.reshape(B, T, -1)
        x = torch.cat((x, x_mark_enc), dim=-1)
        query = self.Linear_query(x)
        key = self.Linear_key(sigma)
        score = torch.einsum("BTD,DK->BTK", query, key.transpose(1, 0)) / torch.sqrt(
            torch.tensor(self.D, dtype=query.dtype, device=query.device)
        )
        score = F.softmax(score, dim=-1)
        return score

    def forward(self, x, x_mark_enc, U, V, sigma):
        residual = x
        x, x_mark_enc = self.t_conv(x, x_mark_enc)
        skip = self.skip_conv(x)
        weight = self.weight_generation(x, sigma, x_mark_enc)
        x = self.s_conv(x, U, V, weight, sigma)
        x = x + residual[:, :, :, -x.size(3):]
        x = self.norm(x)
        return x, skip, x_mark_enc


class Block(nn.ModuleList):
    def __init__(self, block_id, total_t_len, kernel_set, dilation_exp, n_layers,
                 residual_channels, conv_channels, gcn_depth, M, dy_embedding_dim,
                 skip_channels, num_nodes, layer_norm_affline, propalpha, dropout, D, LowRank, mark_dim):
        super().__init__()
        kernel_size = kernel_set[-1]
        if dilation_exp > 1:
            rf_block = int(1 + block_id * (kernel_size - 1) * (dilation_exp ** n_layers - 1) / (dilation_exp - 1))
        else:
            rf_block = block_id * n_layers * (kernel_size - 1) + 1

        dilation_factor = 1
        for i in range(1, n_layers + 1):
            if dilation_exp > 1:
                rf_size_i = int(rf_block + (kernel_size - 1) * (dilation_exp ** i - 1) / (dilation_exp - 1))
            else:
                rf_size_i = rf_block + i * (kernel_size - 1)
            t_len_i = total_t_len - rf_size_i + 1
            self.append(
                Extractor(residual_channels, conv_channels, kernel_set, dilation_factor, gcn_depth, M,
                          dy_embedding_dim, skip_channels, t_len_i, num_nodes, layer_norm_affline,
                          propalpha, dropout, D, LowRank, mark_dim)
            )
            dilation_factor *= dilation_exp

    def forward(self, x, x_mark_enc, U, V, skip_list, sigma):
        for layer in self:
            x, skip, x_mark_enc = layer(x, x_mark_enc, U, V, sigma)
            skip_list.append(skip)
        return x, skip_list


class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        label_len=0,
        features="M",
        input_dim=1,
        output_dim=1,
        residual_channels=16,
        conv_channels=16,
        skip_channels=32,
        end_channels=64,
        dimension=16,
        M=4,
        LowRank=8,
        D=16,
        gcn_depth=2,
        sumba_layers=2,
        layers=2,
        dilation_exponential=1,
        kernel_set=(2, 3, 6, 7),
        propalpha=0.05,
        dropout=0.3,
        layer_norm_affline=True,
        mark_dim=6,
    ):
        super().__init__()
        self.n_blocks = 1
        self.num_nodes = enc_in
        self.dropout = dropout
        self.pred_len = pred_len
        self.seq_length = seq_len
        self.M = M
        self.LowRank = LowRank
        self.output_dim = output_dim
        self.mark_dim = mark_dim
        kernel_set = list(kernel_set)

        self.start_conv = nn.Conv2d(in_channels=input_dim, out_channels=residual_channels, kernel_size=(1, 1))

        kernel_size = 7
        if dilation_exponential > 1:
            self.receptive_field = int(
                1 + (kernel_size - 1) * (dilation_exponential ** layers - 1) / (dilation_exponential - 1)
            )
        else:
            self.receptive_field = layers * (kernel_size - 1) + 1

        self.total_t_len = max(self.receptive_field, self.seq_length)

        self.blocks = nn.ModuleList()
        for block_id in range(self.n_blocks):
            self.blocks.append(
                Block(block_id, self.total_t_len, kernel_set, dilation_exponential, sumba_layers,
                      residual_channels, conv_channels, gcn_depth, M, dimension, skip_channels,
                      self.num_nodes, layer_norm_affline, propalpha, dropout, D, LowRank, mark_dim)
            )

        self.layers = layers

        if self.seq_length > self.receptive_field:
            self.skip0 = nn.Conv2d(in_channels=input_dim, out_channels=skip_channels, kernel_size=(1, self.seq_length), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, self.seq_length - self.receptive_field + 1), bias=True)
        else:
            self.skip0 = nn.Conv2d(in_channels=input_dim, out_channels=skip_channels, kernel_size=(1, self.receptive_field), bias=True)
            self.skipE = nn.Conv2d(in_channels=residual_channels, out_channels=skip_channels, kernel_size=(1, 1), bias=True)

        self.Sigma = nn.Parameter(torch.randn(self.M, self.LowRank), requires_grad=True)
        self.U = nn.Parameter(torch.randn(self.num_nodes, self.LowRank), requires_grad=True)
        self.V = nn.Parameter(torch.randn(self.num_nodes, self.LowRank), requires_grad=True)

        final_channels = pred_len * output_dim
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1), bias=True),
            nn.ReLU(),
            nn.Conv2d(end_channels, final_channels, kernel_size=(1, 1), bias=True),
        )

        nn.init.orthogonal_(self.U)
        nn.init.orthogonal_(self.V)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x_enc: (B, L, N)
        if x_mark_enc is None:
            x_mark_enc = x_enc.new_zeros(x_enc.size(0), x_enc.size(1), self.mark_dim)
        if x_mark_enc.size(-1) != self.mark_dim:
            # Align provided marks to configured mark_dim.
            if x_mark_enc.size(-1) > self.mark_dim:
                x_mark_enc = x_mark_enc[..., : self.mark_dim]
            else:
                pad = self.mark_dim - x_mark_enc.size(-1)
                x_mark_enc = F.pad(x_mark_enc, (0, pad))

        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc / stdev
        B, L, N = x_enc.shape

        inp = x_enc.unsqueeze(-1).transpose(3, 1)  # (B, 1, N, L)
        seq_len = inp.size(3)
        assert seq_len == self.seq_length, "input sequence length not equal to preset sequence length"

        if self.seq_length < self.receptive_field:
            inp = F.pad(inp, (self.receptive_field - self.seq_length, 0, 0, 0))

        x = self.start_conv(inp)
        skip = self.skip0(F.dropout(inp, self.dropout, training=self.training))
        skip_list = [skip]

        for i in range(self.n_blocks):
            x, skip_list = self.blocks[i](x, x_mark_enc, self.U, self.V, skip_list, self.Sigma)

        skip_list.append(self.skipE(x))
        skip_list = torch.cat(skip_list, -1)
        skip_sum = torch.sum(skip_list, dim=3, keepdim=True)
        x = self.out(skip_sum)
        output = x.reshape(B, self.pred_len, -1, N).transpose(-1, -2)
        # output: (B, pred_len, N, output_dim) -> (B, pred_len, N) for output_dim==1
        if output.size(-1) == 1:
            output = output.squeeze(-1)
        else:
            output = output.reshape(B, self.pred_len, -1)

        output = output * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        output = output + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return output[:, -self.pred_len:, :]
