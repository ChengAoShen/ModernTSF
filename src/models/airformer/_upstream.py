"""Upstream AirFormer model ported from CauAir.

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

class AirEmbedding(nn.Module):
    """Embed categorical time variables."""

    def __init__(self):
        super().__init__()
        self.embed_day = nn.Embedding(7, 3)
        self.embed_hour = nn.Embedding(24, 5)

    def forward(self, x):
        x_day = self.embed_day((x[..., -1] * 7).long())
        x_hour = self.embed_hour((x[..., -2] * 24).long())
        out = torch.cat((x_hour, x_day), -1)
        return out


class LatentLayer(nn.Module):
    """Latent layer to compute mean and std."""

    def __init__(self, dm_dim, latent_dim_in, latent_dim_out, hidden_dim, num_layers=2):
        super().__init__()
        self.num_layers = num_layers
        self.enc_in = nn.Sequential(nn.Conv2d(dm_dim + latent_dim_in, hidden_dim, 1))
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, 1))
            layers.append(nn.ReLU(inplace=True))
        self.enc_hidden = nn.Sequential(*layers)
        self.enc_out_1 = nn.Conv2d(hidden_dim, latent_dim_out, 1)
        self.enc_out_2 = nn.Conv2d(hidden_dim, latent_dim_out, 1)

    def forward(self, x):
        h = self.enc_in(x)
        for i in range(self.num_layers):
            h = self.enc_hidden[i](h)
        mu = torch.minimum(self.enc_out_1(h), torch.ones_like(h) * 10)
        sigma = torch.minimum(self.enc_out_2(h), torch.ones_like(h) * 10)
        return mu, sigma


class StochasticModel(nn.Module):
    """Generative / inference model with top-down latent layers."""

    def __init__(self, dm_dim, latent_dim, num_blocks=4):
        super().__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_blocks - 1):
            self.layers.append(LatentLayer(dm_dim, latent_dim, latent_dim, latent_dim, 2))
        self.layers.append(LatentLayer(dm_dim, 0, latent_dim, latent_dim, 2))

    def reparameterize(self, mu, sigma):
        eps = torch.randn_like(sigma, requires_grad=False)
        return mu + eps * sigma

    def forward(self, d):
        _mu, _logsigma = self.layers[-1](d[-1])
        _sigma = torch.exp(_logsigma) + 1e-3
        mus = [_mu]
        sigmas = [_sigma]
        z = [self.reparameterize(_mu, _sigma)]
        for i in reversed(range(len(self.layers) - 1)):
            _mu, _logsigma = self.layers[i](torch.cat((d[i], z[-1]), dim=1))
            _sigma = torch.exp(_logsigma) + 1e-3
            mus.append(_mu)
            sigmas.append(_sigma)
            z.append(self.reparameterize(_mu, _sigma))
        z = torch.stack(z)
        mus = torch.stack(mus)
        sigmas = torch.stack(sigmas)
        return z, mus, sigmas

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TemporalAttention(nn.Module):
    def __init__(self, dim, heads=2, window_size=1, qkv_bias=False,
                 qk_scale=None, dropout=0., causal=True, device=None):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.num_heads = heads
        self.causal = causal
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.window_size = window_size
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B_prev, T_prev, C_prev = x.shape
        if self.window_size > 0:
            x = x.reshape(-1, self.window_size, C_prev)
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if self.causal:
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if self.window_size > 0:
            x = x.reshape(B_prev, T_prev, C_prev)
        return x


class SpatialAttention(nn.Module):
    def __init__(self, dim, heads=4, qkv_bias=False, qk_scale=None,
                 dropout=0., num_sectors=17, assignment=None, mask=None):
        super().__init__()
        assert dim % heads == 0
        self.dim = dim
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.num_sector = num_sectors
        self.assignment = assignment
        self.mask = mask
        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_linear = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.relative_bias = nn.Parameter(torch.randn(heads, 1, num_sectors))
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, C = x.shape
        pre_kv = torch.einsum('bnc,mnr->bmrc', x, self.assignment)
        pre_kv = pre_kv.reshape(-1, self.num_sector, C)
        pre_q = x.reshape(-1, 1, C)
        q = self.q_linear(pre_q).reshape(B * N, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv_linear(pre_kv).reshape(B * N, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.reshape(B, N, self.num_heads, 1, self.num_sector) + self.relative_bias
        mask_val = self.mask.reshape(1, N, 1, 1, self.num_sector)
        attn = attn.masked_fill_(mask_val, float("-inf"))
        attn = attn.reshape(B * N, self.num_heads, 1, self.num_sector).softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B * N, 1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.reshape(B, N, C)
        return x

class CT_MSA(nn.Module):
    """Causal Temporal MSA."""

    def __init__(self, dim, depth, heads, window_size, mlp_dim, num_time, dropout=0., device=None):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_time, dim))
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                TemporalAttention(dim=dim, heads=heads, window_size=window_size,
                                  dropout=dropout, device=device),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        b, c, n, t = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b * n, t, c)
        x = x + self.pos_embedding
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        x = x.reshape(b, n, t, c).permute(0, 3, 1, 2)
        return x


class DS_MSA(nn.Module):
    """Dartboard Spatial MSA."""

    def __init__(self, dim, depth, heads, mlp_dim, num_nodes, dropout=0.,
                 num_sectors=17, assignment=None, mask=None):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                SpatialAttention(dim=dim, heads=heads, dropout=dropout,
                                 num_sectors=num_sectors, assignment=assignment, mask=mask),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        b, c, n, t = x.shape
        x = x.permute(0, 3, 2, 1).reshape(b * t, n, c)
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        x = x.reshape(b, t, n, c).permute(0, 3, 2, 1)
        return x

class AirFormer(nn.Module):
    """The AirFormer model (ported from CauAir)."""

    def __init__(self, node_num, input_dim, output_dim, seq_len, horizon,
                 dropout=0.3, spatial_flag=False, stochastic_flag=True,
                 hidden_channels=32, end_channels=512, blocks=4,
                 mlp_expansion=2, num_heads=2, dartboard=0, device=None):
        super().__init__()
        self.node_num = node_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.horizon = horizon
        self.dropout = dropout
        self.blocks = blocks
        self.spatial_flag = spatial_flag
        self.stochastic_flag = stochastic_flag
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.s_modules = nn.ModuleList()
        self.t_modules = nn.ModuleList()
        self.embedding_air = AirEmbedding()
        self.alpha = 10
        self.device = device if device is not None else torch.device('cpu')

        # Input conv: input_dim + 8 (from AirEmbedding: 5 hour + 3 day)
        self.start_conv = nn.Conv2d(in_channels=input_dim + 8,
                                    out_channels=hidden_channels,
                                    kernel_size=(1, 1))

        for b in range(blocks):
            window_size = self.seq_len // 2 ** (blocks - b - 1)
            self.t_modules.append(CT_MSA(hidden_channels, depth=1, heads=num_heads,
                                         window_size=window_size,
                                         mlp_dim=hidden_channels * mlp_expansion,
                                         num_time=self.seq_len, dropout=dropout,
                                         device=self.device))
            if self.spatial_flag:
                pass  # DS_MSA requires dartboard info not available in generic mode
            else:
                self.residual_convs.append(nn.Conv2d(hidden_channels, hidden_channels, (1, 1)))
            self.bn.append(nn.BatchNorm2d(hidden_channels))

        # Stochastic components
        if self.stochastic_flag:
            self.generative_model = StochasticModel(hidden_channels, hidden_channels, blocks)
            self.inference_model = StochasticModel(hidden_channels, hidden_channels, blocks)
            self.reconstruction_model = nn.Conv2d(blocks * hidden_channels, self.input_dim, (1, 1))
            self.end_conv_1 = nn.Conv2d(hidden_channels * blocks * 2, end_channels, (1, 1))
            self.end_conv_2 = nn.Conv2d(end_channels, self.horizon * self.output_dim, (1, 1))
        else:
            self.end_conv_1 = nn.Conv2d(hidden_channels * blocks, end_channels, (1, 1))
            self.end_conv_2 = nn.Conv2d(end_channels, self.horizon * self.output_dim, (1, 1))

    def forward(self, inputs, label=None, supports=None):
        """
        Parameters
        ----------
        inputs : (B, T, N, F)
        Returns
        -------
        (B, horizon, N, output_dim)
        """
        x = inputs
        # Embed time features
        x_embed = self.embedding_air(x[..., 1:])  # (B, T, N, 8)
        x = torch.cat([x, x_embed], dim=-1)  # (B, T, N, F+8)

        x = x.permute(0, 3, 2, 1)  # (B, C, N, T)
        x = self.start_conv(x)
        d = []
        for i in range(self.blocks):
            if self.spatial_flag:
                x = self.s_modules[i](x)
            else:
                x = self.residual_convs[i](x)
            x = self.t_modules[i](x)
            x = self.bn[i](x)
            d.append(x)

        d = torch.stack(d)

        if self.stochastic_flag:
            d_shift = [(F.pad(d[i], pad=(1, 0))[..., :-1]) for i in range(len(d))]
            d_shift = torch.stack(d_shift)
            z_p, mu_p, sigma_p = self.generative_model(d_shift)
            z_q, mu_q, sigma_q = self.inference_model(d)

            num_blocks, B, C, N, T = d.shape
            z_q = z_q.permute(1, 0, 2, 3, 4).reshape(B, -1, N, T)
            d = d.permute(1, 0, 2, 3, 4).reshape(B, -1, N, T)
            x_hat = torch.cat([d[..., -1:], z_q[..., -1:]], dim=1)
            x_hat = F.relu(self.end_conv_1(x_hat))
            x_hat = self.end_conv_2(x_hat)
        else:
            num_blocks, B, C, N, T = d.shape
            d = d.permute(1, 0, 2, 3, 4).reshape(B, -1, N, T)
            x_hat = F.relu(d[..., -1:])
            x_hat = F.relu(self.end_conv_1(x_hat))
            x_hat = self.end_conv_2(x_hat)

        # x_hat: (B, horizon*output_dim, N, 1) -> (B, horizon, N, output_dim)
        x_hat = x_hat.squeeze(-1)  # (B, horizon*output_dim, N)
        x_hat = x_hat.permute(0, 2, 1)  # (B, N, horizon*output_dim)
        x_hat = x_hat.reshape(B, N, self.horizon, self.output_dim)
        x_hat = x_hat.permute(0, 2, 1, 3)  # (B, horizon, N, output_dim)
        return x_hat
