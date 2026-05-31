"""Verbatim STAEformer model source.

Vendored from CauAir (src/models/staeformer.py).
BaseModel replaced with nn.Module; explicit dimension params added.
"""

import torch
import torch.nn as nn


class AttentionLayer(nn.Module):
    """Multi-head attention without mask."""

    def __init__(self, model_dim, num_heads=8, mask=False):
        super().__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.mask = mask
        self.head_dim = model_dim // num_heads

        self.FC_Q = nn.Linear(model_dim, model_dim)
        self.FC_K = nn.Linear(model_dim, model_dim)
        self.FC_V = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, query, key, value):
        batch_size = query.shape[0]

        query = self.FC_Q(query)
        key = self.FC_K(key)
        value = self.FC_V(value)

        # [num_heads * batch_size, ..., head_dim]
        query = torch.cat(
            torch.split(query, self.head_dim, dim=-1), dim=0)
        key = torch.cat(
            torch.split(key, self.head_dim, dim=-1), dim=0)
        value = torch.cat(
            torch.split(value, self.head_dim, dim=-1), dim=0)

        key = key.transpose(-1, -2)
        attn_score = (query @ key) / self.head_dim**0.5

        if self.mask:
            mask = torch.ones(
                attn_score.shape[-2], attn_score.shape[-1],
                dtype=torch.bool, device=query.device
            ).tril()
            attn_score.masked_fill_(~mask, -torch.inf)

        attn_score = torch.softmax(attn_score, dim=-1)
        out = attn_score @ value


        out = torch.cat(
            torch.split(out, batch_size, dim=0), dim=-1
        )  # (batch_size, ..., tgt_length, head_dim * num_heads)

        out = self.out_proj(out)
        return out


class SelfAttentionLayer(nn.Module):
    def __init__(self, model_dim, feed_forward_dim=2048,
                 num_heads=8, dropout=0, mask=False):
        super().__init__()
        self.attn = AttentionLayer(model_dim, num_heads, mask)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.ln1 = nn.LayerNorm(model_dim)
        self.ln2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim=-2):
        x = x.transpose(dim, -2)
        residual = x
        out = self.attn(x, x, x)
        out = self.dropout1(out)
        out = self.ln1(residual + out)

        residual = out
        out = self.feed_forward(out)
        out = self.dropout2(out)
        out = self.ln2(residual + out)

        out = out.transpose(dim, -2)
        return out


class STAEformer(nn.Module):
    """STAEformer spatiotemporal forecasting model."""

    def __init__(
        self,
        node_num,
        input_dim,
        output_dim,
        seq_len=12,
        horizon=12,
        in_steps=12,
        out_steps=12,
        steps_per_day=24,
        input_embedding_dim=24,
        tod_embedding_dim=24,
        dow_embedding_dim=24,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=56,
        feed_forward_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.1,
        use_mixed_proj=True,
    ):
        super(STAEformer, self).__init__()
        self.node_num = node_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.horizon = horizon
        self.num_nodes = node_num
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)


        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(
                steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)

        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim))
            nn.init.xavier_uniform_(self.node_emb)

        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(
                    in_steps, self.num_nodes, adaptive_embedding_dim)))

        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim)
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, output_dim)

        self.attn_layers_t = nn.ModuleList(
            [
                SelfAttentionLayer(
                    self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )
        self.attn_layers_s = nn.ModuleList(
            [
                SelfAttentionLayer(
                    self.model_dim, feed_forward_dim, num_heads, dropout)
                for _ in range(num_layers)
            ]
        )


    def forward(self, x):
        # x: (B, T, N, C)
        batch_size, in_steps, num_nodes, _ = x.shape

        if self.tod_embedding_dim > 0:
            tod = x[..., 1]  # (B, T, N)
        if self.dow_embedding_dim > 0:
            dow = x[..., 2]  # (B, T, N)
        x = x[..., : self.input_dim]

        x = self.input_proj(x)  # (B, T, N, input_embedding_dim)

        features = [x]
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod[:, :, 0] * self.steps_per_day).long()
            )  # (B, T, tod_embedding_dim)
            tod_emb = tod_emb.unsqueeze(2).expand(
                -1, -1, num_nodes, -1)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                (dow[:, :, 0] * 7).long()
            )  # (B, T, dow_embedding_dim)
            dow_emb = dow_emb.unsqueeze(2).expand(
                -1, -1, num_nodes, -1)
            features.append(dow_emb)
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(
                batch_size, in_steps, -1, -1)
            features.append(spatial_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                batch_size, -1, -1, -1)
            features.append(adp_emb)

        x = torch.cat(features, dim=-1)  # (B, T, N, model_dim)

        for attn_t, attn_s in zip(
            self.attn_layers_t, self.attn_layers_s
        ):
            x = attn_t(x, dim=1)  # temporal
            x = attn_s(x, dim=2)  # spatial

        # (B, T, N, model_dim)
        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (B, N, T, model_dim)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim)
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim)
            out = out.transpose(1, 2)  # (B, out_steps, N, output_dim)
        else:
            out = x.transpose(1, 3)  # (B, model_dim, N, T)
            out = self.temporal_proj(out)  # (B, model_dim, N, out_steps)
            out = self.output_proj(
                out.transpose(1, 3))  # (B, out_steps, N, output_dim)

        return out
