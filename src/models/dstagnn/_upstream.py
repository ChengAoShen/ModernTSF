"""Upstream DSTAGNN model ported from CauAir.

Verbatim logic with BaseModel replaced by nn.Module and explicit parameters.
All helper classes are bundled in this file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ---------------------------------------------------------------------------
# Helper modules
# ---------------------------------------------------------------------------

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, V, attn_mask, res_att):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)
        attn = F.softmax(scores, dim=3)
        context = torch.matmul(attn, V)
        return context, scores


class SScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, Q, K, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)
        return scores


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_head, num_of_d):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_head = n_head
        self.num_of_d = num_of_d
        self.W_Q = nn.Linear(d_model, d_k * n_head, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_head, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_head, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_Q, input_K, input_V, attn_mask, res_att):
        residual, bs = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(bs, self.num_of_d, -1, self.n_head, self.d_k).transpose(2, 3)
        K = self.W_K(input_K).view(bs, self.num_of_d, -1, self.n_head, self.d_k).transpose(2, 3)
        V = self.W_V(input_V).view(bs, self.num_of_d, -1, self.n_head, self.d_v).transpose(2, 3)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)
        context, res_attn = ScaledDotProductAttention(self.d_k)(Q, K, V, attn_mask, res_att)
        context = context.transpose(2, 3).reshape(bs, self.num_of_d, -1, self.n_head * self.d_v)
        output = self.fc(context)
        return self.norm(output), res_attn


class SMultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_head):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.n_head = n_head
        self.W_Q = nn.Linear(d_model, d_k * n_head, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_head, bias=False)

    def forward(self, input_Q, input_K, attn_mask):
        residual, bs = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(bs, -1, self.n_head, self.d_k).transpose(1, 2)
        K = self.W_K(input_K).view(bs, -1, self.n_head, self.d_k).transpose(1, 2)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)
        attn = SScaledDotProductAttention(self.d_k)(Q, K, attn_mask)
        return attn


class Embedding(nn.Module):
    def __init__(self, nb_seq, d_Em, num_of_features, Etype):
        super().__init__()
        self.nb_seq = nb_seq
        self.num_of_features = num_of_features
        self.Etype = Etype
        self.pos_embed = nn.Embedding(nb_seq, d_Em)
        self.norm = nn.LayerNorm(d_Em)

    def forward(self, x, bs):
        device = x.device
        if self.Etype == 'T':
            pos = torch.arange(self.nb_seq, dtype=torch.long, device=device)
            pos = pos.unsqueeze(0).unsqueeze(0).expand(bs, self.num_of_features, self.nb_seq)
            embedding = x.permute(0, 2, 3, 1) + self.pos_embed(pos)
        else:
            pos = torch.arange(self.nb_seq, dtype=torch.long, device=device)
            pos = pos.unsqueeze(0).expand(bs, self.nb_seq)
            embedding = x + self.pos_embed(pos)
        return self.norm(embedding)

class GTU(nn.Module):
    def __init__(self, in_channels, time_stride, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.con2out = nn.Conv2d(in_channels, 2 * in_channels, kernel_size=(1, kernel_size), stride=(1, time_stride))

    def forward(self, x):
        x_causal_conv = self.con2out(x)
        x_p = x_causal_conv[:, :self.in_channels, :, :]
        x_q = x_causal_conv[:, -self.in_channels:, :, :]
        x_gtu = torch.mul(self.tanh(x_p), self.sigmoid(x_q))
        return x_gtu


class cheb_conv_withSAt(nn.Module):
    def __init__(self, K, cheb_polynomials, in_channels, out_channels, node_num):
        super().__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.relu = nn.ReLU(inplace=True)
        self.Theta = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(in_channels, out_channels)) for _ in range(K)])
        self.mask = nn.ParameterList(
            [nn.Parameter(torch.FloatTensor(node_num, node_num)) for _ in range(K)])

    def forward(self, x, spatial_attention, adj_pa):
        bs, node_num, in_channels, seq_len = x.shape
        device = x.device
        outputs = []
        for time_step in range(seq_len):
            graph_signal = x[:, :, :, time_step]
            output = torch.zeros(bs, node_num, self.out_channels, device=device)
            for k in range(self.K):
                T_k = self.cheb_polynomials[k].to(device)
                mask = self.mask[k]
                myspatial_attention = spatial_attention[:, k, :, :] + adj_pa.mul(mask)
                myspatial_attention = F.softmax(myspatial_attention, dim=1)
                T_k_with_at = T_k.mul(myspatial_attention)
                theta_k = self.Theta[k]
                rhs = T_k_with_at.permute(0, 2, 1).matmul(graph_signal)
                output = output + rhs.matmul(theta_k)
            outputs.append(output.unsqueeze(-1))
        return self.relu(torch.cat(outputs, dim=-1))

class DSTAGNN_block(nn.Module):
    def __init__(self, num_of_d, in_channels, K, nb_chev_filter, nb_time_filter,
                 time_stride, cheb_polynomials, adj_pa, node_num, seq_len,
                 d_model, d_k, d_v, n_head):
        super().__init__()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        self.adj_pa = adj_pa
        self.pre_conv = nn.Conv2d(seq_len, d_model, kernel_size=(1, num_of_d))
        self.EmbedT = Embedding(seq_len, node_num, num_of_d, 'T')
        self.EmbedS = Embedding(node_num, d_model, num_of_d, 'S')
        self.TAt = MultiHeadAttention(node_num, d_k, d_v, n_head, num_of_d)
        self.SAt = SMultiHeadAttention(d_model, d_k, d_v, K)
        self.cheb_conv_SAt = cheb_conv_withSAt(K, cheb_polynomials, in_channels, nb_chev_filter, node_num)
        self.gtu3 = GTU(nb_time_filter, time_stride, 3)
        self.gtu5 = GTU(nb_time_filter, time_stride, 5)
        self.gtu7 = GTU(nb_time_filter, time_stride, 7)
        self.pooling = torch.nn.MaxPool2d(kernel_size=(1, 2), stride=None, padding=0)
        self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_stride))
        self.dropout = nn.Dropout(p=0.05)
        self.fcmy = nn.Sequential(nn.Linear(3 * seq_len - 12, seq_len), nn.Dropout(0.05))
        self.ln = nn.LayerNorm(nb_time_filter)

    def forward(self, x, res_att):
        bs, node_num, num_of_features, seq_len = x.shape
        # TAt
        if num_of_features == 1:
            TEmx = self.EmbedT(x, bs)
        else:
            TEmx = x.permute(0, 2, 3, 1)
        TATout, re_At = self.TAt(TEmx, TEmx, TEmx, None, res_att)
        if TATout.shape[-2] == 4:
            TATout = torch.cat([TATout for _ in range(3)], dim=-2)
        x_TAt = self.pre_conv(TATout.permute(0, 2, 3, 1))[:, :, :, -1].permute(0, 2, 1)
        # SAt
        SEmx_TAt = self.EmbedS(x_TAt, bs)
        SEmx_TAt = self.dropout(SEmx_TAt)
        STAt = self.SAt(SEmx_TAt, SEmx_TAt, None)
        # graph convolution
        spatial_gcn = self.cheb_conv_SAt(x, STAt, self.adj_pa)
        # time convolution
        X = spatial_gcn.permute(0, 2, 1, 3)
        x_gtu = []
        x_gtu.append(self.gtu3(X))
        x_gtu.append(self.gtu5(X))
        x_gtu.append(self.gtu7(X))
        time_conv = torch.cat(x_gtu, dim=-1)
        time_conv = self.fcmy(time_conv)
        if num_of_features == 1:
            time_conv_output = self.relu(time_conv)
        else:
            time_conv_output = self.relu(X + time_conv)
        if num_of_features == 1:
            x_residual = self.residual_conv(x.permute(0, 2, 1, 3))
        else:
            x_residual = x.permute(0, 2, 1, 3)
        x_residual = self.ln(F.relu(x_residual[:, -1:] + time_conv_output).permute(0, 3, 2, 1)).permute(0, 2, 3, 1)
        return x_residual, re_At

def compute_cheb_poly(adj_mx, K):
    """Compute Chebyshev polynomials from adjacency matrix."""
    from models._external.graph_utils import normalize_adj_mx
    # Add small perturbation to avoid degenerate Laplacian (e.g., identity adj)
    adj_safe = adj_mx.copy()
    if adj_safe.sum() < 1e-6 or np.allclose(adj_safe, np.eye(adj_safe.shape[0])):
        adj_safe = adj_safe + 1e-4 * np.ones_like(adj_safe)
        np.fill_diagonal(adj_safe, 0)
    L_list = normalize_adj_mx(adj_safe, "scalap", return_type="dense")
    L = L_list[0] if L_list else np.eye(adj_mx.shape[0], dtype=np.float32)
    n = L.shape[0]
    cheb = [np.eye(n, dtype=np.float32)]
    if K > 1:
        cheb.append(L.copy())
    for i in range(2, K):
        cheb.append(2 * L @ cheb[i - 1] - cheb[i - 2])
    return [torch.tensor(c, dtype=torch.float32) for c in cheb]


class DSTAGNN(nn.Module):
    """DSTAGNN: Dynamic Spatial-Temporal Aware Graph Neural Network."""

    def __init__(self, adj_mx, node_num, input_dim, output_dim, seq_len, horizon,
                 d_model=64, d_k=8, d_v=8, n_heads=4):
        super().__init__()
        self.node_num = node_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.horizon = horizon

        order = n_heads  # K for Chebyshev
        nb_block = 2
        nb_chev_filter = input_dim
        nb_time_filter = input_dim
        time_stride = 1

        cheb_polynomials = compute_cheb_poly(adj_mx, order)
        adj_pa = torch.tensor(adj_mx, dtype=torch.float32)

        self.BlockList = nn.ModuleList([
            DSTAGNN_block(input_dim, input_dim, order, nb_chev_filter, nb_time_filter,
                          time_stride, cheb_polynomials, adj_pa, node_num, seq_len,
                          d_model, d_k, d_v, n_heads)
        ])
        self.BlockList.extend([
            DSTAGNN_block(input_dim * nb_time_filter, nb_chev_filter, order,
                          nb_chev_filter, nb_time_filter, 1, cheb_polynomials,
                          adj_pa, node_num, seq_len // time_stride, d_model, d_k, d_v, n_heads)
            for _ in range(nb_block - 1)
        ])
        self.final_conv = nn.Conv2d(
            int((seq_len / time_stride) * nb_block), 128,
            kernel_size=(1, nb_time_filter))
        self.final_fc = nn.Linear(128, horizon)

    def forward(self, x, label=None):
        """
        Parameters
        ----------
        x : (B, T, N, F)
        Returns
        -------
        (B, horizon, N, output_dim)
        """
        x = x.permute(0, 2, 3, 1)  # (B, N, F, T)
        need_concat = []
        res_att = 0
        for block in self.BlockList:
            x, res_att = block(x, res_att)
            need_concat.append(x)
        final_x = torch.cat(need_concat, dim=-1)
        output = self.final_conv(final_x.permute(0, 3, 1, 2))[:, :, :, -1].permute(0, 2, 1)
        output = self.final_fc(output)
        output = output.unsqueeze(-1).transpose(1, 2)
        return output
