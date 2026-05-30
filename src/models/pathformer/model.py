"""Pathformer model implementation.

Vendored/adapted from https://github.com/decisionintelligence/pathformer
(models/PathFormer.py, layers/AMS.py, layers/Layer.py, layers/Embedding.py,
layers/RevIN.py, utils/Other.py). The upstream repository ships no LICENSE
file (license: NOASSERTION / none declared); it is the official code release
for the ICLR 2024 paper "Pathformer: Multi-scale Transformers with Adaptive
Pathways for Time Series Forecasting".

Adapted for ModernTSF: the upstream ``configs``-object constructor is replaced
with plain keyword arguments, the hard-coded ``cuda`` device is removed (the
mixture-of-experts router infers its device from the input tensors), and only
the long-term-forecast path is kept. ``forward`` follows the ModernTSF contract
``forward(x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None)`` and
returns ``(B, pred_len, c_out)``; the auxiliary MoE balance loss produced
internally is discarded (the trainer consumes only the forecast tensor).

The Pathformer-specific blocks (adaptive multi-scale router ``AMS``, the
multi-scale ``Transformer_Layer`` with intra-/inter-patch attention, the
factorized ``WeightGenerator``, the Fourier/decomposition helpers, the sparse
MoE dispatcher and the RevIN layer) are vendored verbatim below because they do
not match the shared ``models.module.*`` layers.
"""

from __future__ import annotations

import math
from operator import mul
from functools import reduce

import numpy as np
import torch
import torch.fft as fft
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce as ein_reduce, repeat
from torch.distributions.normal import Normal
from torch.nn import init


# --------------------------------------------------------------------------- #
# RevIN
# --------------------------------------------------------------------------- #
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


# --------------------------------------------------------------------------- #
# Positional encoding helpers (from layers/Embedding.py)
# --------------------------------------------------------------------------- #
def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


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
    elif pe == "sincos":
        W_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else:
        raise ValueError(f"{pe} is not a valid pe")
    return nn.Parameter(W_pos, requires_grad=learn_pe)


# --------------------------------------------------------------------------- #
# Decomposition / Fourier / MoE helpers (from utils/Other.py)
# --------------------------------------------------------------------------- #
class SparseDispatcher(object):
    def __init__(self, num_experts, gates):
        self._gates = gates
        self._num_experts = num_experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        _, self._expert_index = sorted_experts.split(1, dim=1)
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1], 0]
        self._part_sizes = (gates > 0).sum(0).tolist()
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)

    def dispatch(self, inp):
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)

    def combine(self, expert_out, multiply_by_gates=True):
        stitched = torch.cat(expert_out, 0).exp()
        if multiply_by_gates:
            stitched = torch.einsum("ijkh,ik -> ijkh", stitched, self._nonzero_gates)
        zeros = torch.zeros(
            self._gates.size(0),
            expert_out[-1].size(1),
            expert_out[-1].size(2),
            expert_out[-1].size(3),
            requires_grad=True,
            device=stitched.device,
        )
        combined = zeros.index_add(0, self._batch_index, stitched.float())
        combined[combined == 0] = np.finfo(float).eps
        return combined.log()

    def expert_to_gates(self):
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)


class MLP(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Conv2d(
            in_channels=input_size, out_channels=output_size, kernel_size=(1, 1), bias=True
        )

    def forward(self, x):
        return self.fc(x)


class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(
            1, self.kernel_size - 1 - math.floor((self.kernel_size - 1) // 2), 1
        )
        end = x[:, -1:, :].repeat(1, math.floor((self.kernel_size - 1) // 2), 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp_multi(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.moving_avg = [moving_avg(kernel, stride=1) for kernel in kernel_size]
        self.layer = torch.nn.Linear(1, len(kernel_size))

    def forward(self, x):
        moving_mean = []
        for func in self.moving_avg:
            avg = func(x)
            moving_mean.append(avg.unsqueeze(-1))
        moving_mean = torch.cat(moving_mean, dim=-1)
        moving_mean = torch.sum(
            moving_mean * nn.Softmax(-1)(self.layer(x.unsqueeze(-1))), dim=-1
        )
        res = x - moving_mean
        return res, moving_mean


class FourierLayer(nn.Module):
    def __init__(self, pred_len, k=None, low_freq=1, output_attention=False):
        super().__init__()
        self.pred_len = pred_len
        self.k = k
        self.low_freq = low_freq
        self.output_attention = output_attention

    def forward(self, x):
        """x: (b, t, d)"""
        b, t, d = x.shape
        x_freq = fft.rfft(x, dim=1)

        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq : -1]
            f = fft.rfftfreq(t)[self.low_freq : -1]
        else:
            x_freq = x_freq[:, self.low_freq :]
            f = fft.rfftfreq(t)[self.low_freq :]

        x_freq, index_tuple = self.topk_freq(x_freq)
        f = repeat(f, "f -> b f d", b=x_freq.size(0), d=x_freq.size(2))
        f = f.to(x_freq.device)
        f = rearrange(f[index_tuple], "b f d -> b f () d").to(x_freq.device)

        return self.extrapolate(x_freq, f, t), None

    def extrapolate(self, x_freq, f, t):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t_val = rearrange(
            torch.arange(t + self.pred_len, dtype=torch.float), "t -> () () t ()"
        ).to(x_freq.device)

        amp = rearrange(x_freq.abs() / t, "b f d -> b f () d")
        phase = rearrange(x_freq.angle(), "b f d -> b f () d")

        x_time = amp * torch.cos(2 * math.pi * f * t_val + phase)

        return ein_reduce(x_time, "b f t d -> b t d", "sum")

    def topk_freq(self, x_freq):
        values, indices = torch.topk(x_freq.abs(), self.k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(
            torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)), indexing="ij"
        )
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]
        return x_freq, index_tuple


# --------------------------------------------------------------------------- #
# Multi-scale transformer layer (from layers/Layer.py)
# --------------------------------------------------------------------------- #
class CustomLinear(nn.Module):
    def __init__(self, factorized):
        super().__init__()
        self.factorized = factorized

    def forward(self, input, weights, biases):
        if self.factorized:
            return torch.matmul(input.unsqueeze(3), weights).squeeze(3) + biases
        return torch.matmul(input, weights) + biases


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        return x.transpose(*self.dims)


class WeightGenerator(nn.Module):
    def __init__(self, in_dim, out_dim, mem_dim, num_nodes, factorized, number_of_weights=4):
        super().__init__()
        self.number_of_weights = number_of_weights
        self.mem_dim = mem_dim
        self.num_nodes = num_nodes
        self.factorized = factorized
        self.out_dim = out_dim
        if self.factorized:
            self.memory = nn.Parameter(torch.randn(num_nodes, mem_dim), requires_grad=True)
            self.generator = nn.Sequential(
                *[
                    nn.Linear(mem_dim, 64),
                    nn.Tanh(),
                    nn.Linear(64, 64),
                    nn.Tanh(),
                    nn.Linear(64, 100),
                ]
            )
            self.mem_dim = 10
            self.P = nn.ParameterList(
                [
                    nn.Parameter(torch.Tensor(in_dim, self.mem_dim), requires_grad=True)
                    for _ in range(number_of_weights)
                ]
            )
            self.Q = nn.ParameterList(
                [
                    nn.Parameter(torch.Tensor(self.mem_dim, out_dim), requires_grad=True)
                    for _ in range(number_of_weights)
                ]
            )
            self.B = nn.ParameterList(
                [
                    nn.Parameter(
                        torch.Tensor(self.mem_dim**2, out_dim), requires_grad=True
                    )
                    for _ in range(number_of_weights)
                ]
            )
        else:
            self.P = nn.ParameterList(
                [
                    nn.Parameter(torch.Tensor(in_dim, out_dim), requires_grad=True)
                    for _ in range(number_of_weights)
                ]
            )
            self.B = nn.ParameterList(
                [
                    nn.Parameter(torch.Tensor(1, out_dim), requires_grad=True)
                    for _ in range(number_of_weights)
                ]
            )
        self.reset_parameters()

    def reset_parameters(self):
        list_params = [self.P, self.Q, self.B] if self.factorized else [self.P]
        for weight_list in list_params:
            for weight in weight_list:
                init.kaiming_uniform_(weight, a=math.sqrt(5))

        if not self.factorized:
            for i in range(self.number_of_weights):
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.P[i])
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(self.B[i], -bound, bound)

    def forward(self):
        if self.factorized:
            memory = self.generator(self.memory.unsqueeze(1))
            bias = [
                torch.matmul(memory, self.B[i]).squeeze(1)
                for i in range(self.number_of_weights)
            ]
            memory = memory.view(self.num_nodes, self.mem_dim, self.mem_dim)
            weights = [
                torch.matmul(torch.matmul(self.P[i], memory), self.Q[i])
                for i in range(self.number_of_weights)
            ]
            return weights, bias
        return self.P, self.B


class Intra_Patch_Attention(nn.Module):
    def __init__(self, d_model, factorized):
        super().__init__()
        self.head = 2
        if d_model % self.head != 0:
            raise Exception("Hidden size is not divisible by the number of attention heads")
        self.head_size = int(d_model // self.head)
        self.custom_linear = CustomLinear(factorized)

    def forward(
        self,
        query,
        key,
        value,
        weights_distinct,
        biases_distinct,
        weights_shared,
        biases_shared,
    ):
        batch_size = query.shape[0]

        key = self.custom_linear(key, weights_distinct[0], biases_distinct[0])
        value = self.custom_linear(value, weights_distinct[1], biases_distinct[1])
        query = torch.cat(torch.split(query, self.head_size, dim=-1), dim=0)
        key = torch.cat(torch.split(key, self.head_size, dim=-1), dim=0)
        value = torch.cat(torch.split(value, self.head_size, dim=-1), dim=0)

        query = query.permute((0, 2, 1, 3))
        key = key.permute((0, 2, 3, 1))
        value = value.permute((0, 2, 1, 3))

        attention = torch.matmul(query, key)
        attention /= self.head_size**0.5
        attention = torch.softmax(attention, dim=-1)

        x = torch.matmul(attention, value)
        x = x.permute((0, 2, 1, 3))
        x = torch.cat(torch.split(x, batch_size, dim=0), dim=-1)

        if x.shape[0] == 0:
            x = x.repeat(1, 1, 1, int(weights_shared[0].shape[-1] / x.shape[-1]))

        x = self.custom_linear(x, weights_shared[0], biases_shared[0])
        x = torch.relu(x)
        x = self.custom_linear(x, weights_shared[1], biases_shared[1])
        return x, attention


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, n_heads, attn_dropout=0.0, res_attention=False, lsa=False):
        super().__init__()
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_attention = res_attention
        head_dim = d_model // n_heads
        self.scale = nn.Parameter(torch.tensor(head_dim**-0.5), requires_grad=lsa)
        self.lsa = lsa

    def forward(self, q, k, v, prev=None, key_padding_mask=None, attn_mask=None):
        attn_scores = torch.matmul(q, k) * self.scale
        if prev is not None:
            attn_scores = attn_scores + prev
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_scores.masked_fill_(attn_mask, -np.inf)
            else:
                attn_scores += attn_mask
        if key_padding_mask is not None:
            attn_scores.masked_fill_(
                key_padding_mask.unsqueeze(1).unsqueeze(2), -np.inf
            )
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        output = torch.matmul(attn_weights, v)
        return output, attn_weights


class Inter_Patch_Attention(nn.Module):
    def __init__(
        self,
        d_model,
        out_dim,
        n_heads,
        d_k=None,
        d_v=None,
        res_attention=False,
        attn_dropout=0.0,
        proj_dropout=0.0,
        qkv_bias=True,
        lsa=False,
    ):
        super().__init__()
        d_k = d_model // n_heads if d_k is None else d_k
        d_v = d_model // n_heads if d_v is None else d_v
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v

        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=qkv_bias)

        self.res_attention = res_attention
        self.sdp_attn = ScaledDotProductAttention(
            d_model, n_heads, attn_dropout=attn_dropout, res_attention=self.res_attention, lsa=lsa
        )
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_v, out_dim), nn.Dropout(proj_dropout))

    def forward(self, Q, K=None, V=None, prev=None, key_padding_mask=None, attn_mask=None):
        bs = Q.size(0)
        if K is None:
            K = Q
        if V is None:
            V = Q

        q_s = self.W_Q(Q).view(bs, Q.shape[1], self.n_heads, self.d_k).transpose(1, 2)
        k_s = self.W_K(K).view(bs, K.shape[1], self.n_heads, self.d_k).permute(0, 2, 3, 1)
        v_s = self.W_V(V).view(bs, V.shape[1], self.n_heads, self.d_v).transpose(1, 2)

        if self.res_attention:
            output, attn_weights, attn_scores = self.sdp_attn(
                q_s, k_s, v_s, prev=prev, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
        else:
            output, attn_weights = self.sdp_attn(
                q_s, k_s, v_s, key_padding_mask=key_padding_mask, attn_mask=attn_mask
            )
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(bs, Q.shape[1], self.n_heads * self.d_v)
        )
        output = self.to_out(output)
        return output, attn_weights


class Transformer_Layer(nn.Module):
    def __init__(
        self,
        device,
        d_model,
        d_ff,
        num_nodes,
        patch_nums,
        patch_size,
        dynamic,
        factorized,
        layer_number,
        batch_norm,
    ):
        super().__init__()
        self.device = device
        self.d_model = d_model
        self.num_nodes = num_nodes
        self.dynamic = dynamic
        self.patch_nums = patch_nums
        self.patch_size = patch_size
        self.layer_number = layer_number
        self.batch_norm = batch_norm

        # intra_patch_attention
        self.intra_embeddings = nn.Parameter(
            torch.rand(self.patch_nums, 1, 1, self.num_nodes, 16), requires_grad=True
        )
        self.embeddings_generator = nn.ModuleList(
            [nn.Sequential(*[nn.Linear(16, self.d_model)]) for _ in range(self.patch_nums)]
        )
        self.intra_d_model = self.d_model
        self.intra_patch_attention = Intra_Patch_Attention(
            self.intra_d_model, factorized=factorized
        )
        self.weights_generator_distinct = WeightGenerator(
            self.intra_d_model,
            self.intra_d_model,
            mem_dim=16,
            num_nodes=num_nodes,
            factorized=factorized,
            number_of_weights=2,
        )
        self.weights_generator_shared = WeightGenerator(
            self.intra_d_model,
            self.intra_d_model,
            mem_dim=None,
            num_nodes=num_nodes,
            factorized=False,
            number_of_weights=2,
        )
        self.intra_Linear = nn.Linear(self.patch_nums, self.patch_nums * self.patch_size)

        # inter_patch_attention
        self.stride = patch_size
        self.inter_d_model = self.d_model * self.patch_size
        self.emb_linear = nn.Linear(self.inter_d_model, self.inter_d_model)
        self.W_pos = positional_encoding(
            pe="zeros", learn_pe=True, q_len=self.patch_nums, d_model=self.inter_d_model
        )
        n_heads = self.d_model
        d_k = self.inter_d_model // n_heads
        d_v = self.inter_d_model // n_heads
        self.inter_patch_attention = Inter_Patch_Attention(
            self.inter_d_model,
            self.inter_d_model,
            n_heads,
            d_k,
            d_v,
            attn_dropout=0,
            proj_dropout=0.1,
            res_attention=False,
        )

        # Normalization
        self.norm_attn = nn.Sequential(
            Transpose(1, 2), nn.BatchNorm1d(self.d_model), Transpose(1, 2)
        )
        self.norm_ffn = nn.Sequential(
            Transpose(1, 2), nn.BatchNorm1d(self.d_model), Transpose(1, 2)
        )

        # FFN
        self.d_ff = d_ff
        self.dropout = nn.Dropout(0.1)
        self.ff = nn.Sequential(
            nn.Linear(self.d_model, self.d_ff, bias=True),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(self.d_ff, self.d_model, bias=True),
        )

    def forward(self, x):
        new_x = x
        batch_size = x.size(0)
        intra_out_concat = None

        weights_shared, biases_shared = self.weights_generator_shared()
        weights_distinct, biases_distinct = self.weights_generator_distinct()

        # intra Attention
        for i in range(self.patch_nums):
            t = x[:, i * self.patch_size : (i + 1) * self.patch_size, :, :]

            intra_emb = self.embeddings_generator[i](self.intra_embeddings[i]).expand(
                batch_size, -1, -1, -1
            )
            t = torch.cat([intra_emb, t], dim=1)
            out, attention = self.intra_patch_attention(
                intra_emb,
                t,
                t,
                weights_distinct,
                biases_distinct,
                weights_shared,
                biases_shared,
            )

            if intra_out_concat is None:
                intra_out_concat = out
            else:
                intra_out_concat = torch.cat([intra_out_concat, out], dim=1)

        intra_out_concat = intra_out_concat.permute(0, 3, 2, 1)
        intra_out_concat = self.intra_Linear(intra_out_concat)
        intra_out_concat = intra_out_concat.permute(0, 3, 2, 1)

        # inter Attention
        x = x.unfold(dimension=1, size=self.patch_size, step=self.stride)
        x = x.permute(0, 2, 1, 3, 4)
        b, nvar, patch_num, dim, patch_len = x.shape

        x = torch.reshape(
            x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3] * x.shape[-1])
        )

        x = self.emb_linear(x)
        x = self.dropout(x + self.W_pos)

        inter_out, attention = self.inter_patch_attention(Q=x, K=x, V=x)
        inter_out = torch.reshape(
            inter_out, (b, nvar, inter_out.shape[-2], inter_out.shape[-1])
        )
        inter_out = torch.reshape(
            inter_out, (b, nvar, inter_out.shape[-2], self.patch_size, self.d_model)
        )
        inter_out = torch.reshape(
            inter_out, (b, self.patch_size * self.patch_nums, nvar, self.d_model)
        )

        out = new_x + intra_out_concat + inter_out
        if self.batch_norm:
            out = self.norm_attn(
                out.reshape(b * nvar, self.patch_size * self.patch_nums, self.d_model)
            )
        out = self.dropout(out)
        out = self.ff(out) + out
        if self.batch_norm:
            out = self.norm_ffn(out).reshape(
                b, self.patch_size * self.patch_nums, nvar, self.d_model
            )
        return out, attention


# --------------------------------------------------------------------------- #
# Adaptive multi-scale router (from layers/AMS.py)
# --------------------------------------------------------------------------- #
class AMS(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        num_experts,
        device,
        num_nodes=1,
        d_model=32,
        d_ff=64,
        dynamic=False,
        patch_size=[8, 6, 4, 2],
        noisy_gating=True,
        k=4,
        layer_number=1,
        residual_connection=1,
        batch_norm=False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.output_size = output_size
        self.input_size = input_size
        self.k = k

        self.start_linear = nn.Linear(in_features=num_nodes, out_features=1)
        self.seasonality_model = FourierLayer(pred_len=0, k=3)
        self.trend_model = series_decomp_multi(kernel_size=[4, 8, 12])

        self.experts = nn.ModuleList()
        self.MLPs = nn.ModuleList()
        for patch in patch_size:
            patch_nums = int(input_size / patch)
            self.experts.append(
                Transformer_Layer(
                    device=device,
                    d_model=d_model,
                    d_ff=d_ff,
                    dynamic=dynamic,
                    num_nodes=num_nodes,
                    patch_nums=patch_nums,
                    patch_size=patch,
                    factorized=True,
                    layer_number=layer_number,
                    batch_norm=batch_norm,
                )
            )

        self.w_noise = nn.Linear(input_size, num_experts)
        self.w_gate = nn.Linear(input_size, num_experts)

        self.residual_connection = residual_connection
        self.end_MLP = MLP(input_size=input_size, output_size=output_size)

        self.noisy_gating = noisy_gating
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert self.k <= self.num_experts

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = (
            torch.arange(batch, device=clean_values.device) * m + self.k
        )
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
        )
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
        )
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def seasonality_and_trend_decompose(self, x):
        x = x[:, :, :, 0]
        _, trend = self.trend_model(x)
        seasonality, _ = self.seasonality_model(x)
        return x + seasonality + trend

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        x = self.start_linear(x).squeeze(-1)

        clean_logits = self.w_gate(x)
        if self.noisy_gating and train:
            raw_noise_stddev = self.w_noise(x)
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        top_logits, top_indices = logits.topk(
            min(self.k + 1, self.num_experts), dim=1
        )
        top_k_logits = top_logits[:, : self.k]
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (
                self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits)
            ).sum(0)
        else:
            load = self._gates_to_load(gates)
        return gates, load

    def forward(self, x, loss_coef=1e-2):
        new_x = self.seasonality_and_trend_decompose(x)

        gates, load = self.noisy_top_k_gating(new_x, self.training)
        importance = gates.sum(0)
        balance_loss = self.cv_squared(importance) + self.cv_squared(load)
        balance_loss *= loss_coef
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        expert_outputs = [
            self.experts[i](expert_inputs[i])[0] for i in range(self.num_experts)
        ]
        output = dispatcher.combine(expert_outputs)
        if self.residual_connection:
            output = output + x
        return output, balance_loss


# --------------------------------------------------------------------------- #
# Pathformer (from models/PathFormer.py)
# --------------------------------------------------------------------------- #
class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        features="M",
        layer_nums=2,
        k=2,
        num_experts=4,
        patch_size_list=None,
        d_model=16,
        d_ff=64,
        residual_connection=1,
        revin=True,
        batch_norm=False,
        **kwargs,
    ):
        super().__init__()
        self.layer_nums = layer_nums
        self.num_nodes = enc_in
        self.pre_len = pred_len
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features
        self.k = k
        self.d_model = d_model
        self.d_ff = d_ff
        self.residual_connection = residual_connection
        self.revin = revin

        # num_experts per layer
        self.num_experts_list = [num_experts] * layer_nums

        # patch_size_list: flat list reshaped to (layer_nums, num_experts).
        if patch_size_list is None:
            patch_size_list = [16, 12, 8, 6] * layer_nums
        flat = list(patch_size_list)
        expected = layer_nums * num_experts
        if len(flat) != expected:
            raise ValueError(
                f"patch_size_list length {len(flat)} != layer_nums*num_experts "
                f"({layer_nums}*{num_experts}={expected})"
            )
        self.patch_size_list = (
            np.array(flat).reshape(layer_nums, num_experts).tolist()
        )

        if self.revin:
            self.revin_layer = RevIN(
                num_features=self.num_nodes, affine=False, subtract_last=False
            )

        self.start_fc = nn.Linear(in_features=1, out_features=self.d_model)
        self.AMS_lists = nn.ModuleList()
        self.batch_norm = batch_norm

        # Device is inferred from input tensors at forward time; pass None here.
        for num in range(self.layer_nums):
            self.AMS_lists.append(
                AMS(
                    self.seq_len,
                    self.seq_len,
                    self.num_experts_list[num],
                    None,
                    k=self.k,
                    num_nodes=self.num_nodes,
                    patch_size=self.patch_size_list[num],
                    noisy_gating=True,
                    d_model=self.d_model,
                    d_ff=self.d_ff,
                    layer_number=num + 1,
                    residual_connection=self.residual_connection,
                    batch_norm=self.batch_norm,
                )
            )
        self.projections = nn.Sequential(
            nn.Linear(self.seq_len * self.d_model, self.pre_len)
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        x = x_enc
        balance_loss = 0
        if self.revin:
            x = self.revin_layer(x, "norm")
        out = self.start_fc(x.unsqueeze(-1))

        batch_size = x.shape[0]

        for layer in self.AMS_lists:
            out, aux_loss = layer(out)
            balance_loss = balance_loss + aux_loss

        out = out.permute(0, 2, 1, 3).reshape(batch_size, self.num_nodes, -1)
        out = self.projections(out).transpose(2, 1)

        if self.revin:
            out = self.revin_layer(out, "denorm")

        return out[:, -self.pred_len :, :]
