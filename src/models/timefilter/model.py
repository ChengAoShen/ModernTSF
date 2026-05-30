"""TimeFilter model implementation.

Vendored/adapted from https://github.com/TROUBADOUR000/TimeFilter
(models/TimeFilter.py and layers/TimeFilter_layers.py). The upstream
repository ships no LICENSE file (GitHub reports ``license: null``); its
README acknowledges Time-Series-Library (MIT) and iTransformer (MIT) as the
codebases it derives from. No copyleft (GPL/AGPL) license applies.

TimeFilter: Patch-Specific Spatial-Temporal Graph Filtration for Time Series
Forecasting (ICML 2025), Yifan Hu et al.

Adapted for ModernTSF:
- The upstream ``configs``-object constructor is replaced with plain keyword
  arguments.
- The shared ``PositionalEmbedding`` and ``Normalize`` layers under
  ``models.module.*`` are reused.
- The patch-graph backbone (``GraphLearner`` / ``mask_moe`` / ``GraphFilter`` /
  ``GraphBlock`` / ``TimeFilter_Backbone``) learns a patch graph INTERNALLY
  (no external adjacency); it is vendored locally below.
- The patch-region masks (S / T / ST partition of the patch-graph nodes) are
  precomputed inside ``Model`` from ``seq_len``/``patch_len``/``n_vars`` and
  registered as a buffer, so ``forward`` needs no external ``masks`` argument.
- The forward contract is ``(x_enc, x_mark_enc, x_dec, x_mark_dec)`` returning
  ``(B, pred_len, c_out)``. The upstream MoE auxiliary loss cannot be routed
  through this trainer interface, so it is computed and stored on
  ``self.last_moe_loss`` for inspection but not returned.
- Only the long-term forecasting path is kept (classification / imputation /
  anomaly branches dropped).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.module.embed import PositionalEmbedding
from models.module.standard_norm import Normalize


class PatchEmbed(nn.Module):
    def __init__(self, dim, patch_len, stride=None, pos=True):
        super().__init__()
        self.patch_len = patch_len
        self.stride = patch_len if stride is None else stride
        self.patch_proj = nn.Linear(self.patch_len, dim)
        self.pos = pos
        if self.pos:
            self.pe = PositionalEmbedding(dim)

    def forward(self, x):
        # x: [B, C*T]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = self.patch_proj(x)  # [B, L, D]
        if self.pos:
            x = x + self.pe(x)
        return x


class GCN(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.n_heads = n_heads

    def forward(self, adj, x):
        # adj [B, H, L, L]
        B, L, D = x.shape
        x = self.proj(x).view(B, L, self.n_heads, -1)  # [B, L, H, D_]
        adj = F.normalize(adj, p=1, dim=-1)
        x = torch.einsum("bhij,bjhd->bihd", adj, x).contiguous()  # [B, L, H, D_]
        x = x.view(B, L, -1)
        return x


def mask_topk(x, alpha=0.5, largest=False):
    # x: [B, H, L, L]
    k = int(alpha * x.shape[-1])
    if k <= 0:
        return torch.ones_like(x, dtype=torch.float32)
    _, topk_indices = torch.topk(x, k, dim=-1, largest=largest)
    mask = torch.ones_like(x, dtype=torch.float32)
    mask.scatter_(-1, topk_indices, 0)  # 0 marks the selected topk
    return mask  # [B, H, L, L]


class mask_moe(nn.Module):
    def __init__(self, n_vars, top_p=0.5, num_experts=3, in_dim=96):
        super().__init__()
        self.num_experts = num_experts
        self.n_vars = n_vars
        self.in_dim = in_dim

        self.gate = nn.Linear(self.in_dim, num_experts, bias=False)
        self.noise = nn.Linear(self.in_dim, num_experts, bias=False)
        self.noisy_gating = True
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(2)
        self.top_p = top_p

    def cv_squared(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def cross_entropy(self, x):
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return -torch.mul(x, torch.log(x + eps)).sum(dim=1).mean()

    def noisy_top_k_gating(self, x, is_training, noise_epsilon=1e-2):
        clean_logits = self.gate(x)
        if self.noisy_gating and is_training:
            raw_noise = self.noise(x)
            noise_stddev = self.softplus(raw_noise) + noise_epsilon
            noisy_logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
            logits = noisy_logits
        else:
            logits = clean_logits

        logits = self.softmax(logits)
        loss_dynamic = self.cross_entropy(logits)

        sorted_probs, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs > self.top_p

        threshold_indices = mask.long().argmax(dim=-1)
        threshold_mask = F.one_hot(
            threshold_indices, num_classes=sorted_indices.size(-1)
        ).bool()
        mask = mask & ~threshold_mask

        top_p_mask = torch.zeros_like(mask)
        zero_indices = (mask == 0).nonzero(as_tuple=True)
        top_p_mask[
            zero_indices[0],
            zero_indices[1],
            sorted_indices[zero_indices[0], zero_indices[1], zero_indices[2]],
        ] = 1

        sorted_probs = torch.where(mask, 0.0, sorted_probs)
        loss_importance = self.cv_squared(sorted_probs.sum(0))
        lambda_2 = 0.1
        loss = loss_importance + lambda_2 * loss_dynamic

        return top_p_mask, loss

    def forward(self, x, masks, is_training=False):
        # x [B, H, L, L]
        B, H, L, _ = x.shape
        device = x.device
        dtype = torch.float32

        mask_base = torch.eye(L, device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
        if self.top_p == 0.0:
            return mask_base, torch.zeros((), device=device, dtype=dtype)

        x = x.reshape(B * H, L, L)
        gates, loss = self.noisy_top_k_gating(x, is_training)
        gates = gates.reshape(B, H, L, -1).float()  # [B, H, L, 3]

        # masks: [L, 3, L]
        mask = torch.einsum("bhld,lds->bhls", gates, masks) + mask_base
        return mask, loss


class GraphLearner(nn.Module):
    def __init__(self, dim, n_vars, top_p=0.5, in_dim=96):
        super().__init__()
        self.dim = dim
        self.proj_1 = nn.Linear(dim, dim)
        self.proj_2 = nn.Linear(dim, dim)
        self.n_vars = n_vars
        self.mask_moe = mask_moe(n_vars, top_p=top_p, in_dim=in_dim)

    def forward(self, x, masks, alpha=0.5, is_training=False):
        # x: [B, H, L, D]
        adj = F.gelu(
            torch.einsum("bhid,bhjd->bhij", self.proj_1(x), self.proj_2(x))
        )
        adj = adj * mask_topk(adj, alpha)  # KNN
        mask, loss = self.mask_moe(adj, masks, is_training)
        adj = adj * mask
        return adj, loss  # [B, H, L, L]


class GraphFilter(nn.Module):
    def __init__(self, dim, n_vars, n_heads=4, scale=None, top_p=0.5, dropout=0.0, in_dim=96):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.scale = dim ** (-0.5) if scale is None else scale
        self.dropout = nn.Dropout(dropout)
        self.graph_learner = GraphLearner(self.dim // self.n_heads, n_vars, top_p, in_dim=in_dim)
        self.graph_conv = GCN(self.dim, self.n_heads)

    def forward(self, x, masks, alpha=0.5, is_training=False):
        # x: [B, L, D]
        B, L, D = x.shape
        adj, loss = self.graph_learner(
            x.reshape(B, L, self.n_heads, -1).permute(0, 2, 1, 3),
            masks,
            alpha,
            is_training,
        )  # [B, H, L, L]
        adj = torch.softmax(adj, dim=-1)
        adj = self.dropout(adj)
        out = self.graph_conv(adj, x)
        return out, loss  # [B, L, D]


class GraphBlock(nn.Module):
    def __init__(self, dim, n_vars, d_ff=None, n_heads=4, top_p=0.5, dropout=0.0, in_dim=96):
        super().__init__()
        self.dim = dim
        self.d_ff = dim * 4 if d_ff is None else d_ff
        self.gnn = GraphFilter(self.dim, n_vars, n_heads, top_p=top_p, dropout=dropout, in_dim=in_dim)
        self.norm1 = nn.LayerNorm(self.dim)
        self.ffn = nn.Sequential(
            nn.Linear(self.dim, self.d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_ff, self.dim),
        )
        self.norm2 = nn.LayerNorm(self.dim)

    def forward(self, x, masks, alpha=0.5, is_training=False):
        out, loss = self.gnn(self.norm1(x), masks, alpha, is_training)
        x = x + out
        x = x + self.ffn(self.norm2(x))
        return x, loss


class TimeFilter_Backbone(nn.Module):
    def __init__(self, hidden_dim, n_vars, d_ff=None, n_heads=4, n_blocks=3, top_p=0.5, dropout=0.0, in_dim=96):
        super().__init__()
        self.dim = hidden_dim
        self.d_ff = self.dim * 2 if d_ff is None else d_ff
        self.blocks = nn.ModuleList(
            [
                GraphBlock(self.dim, n_vars, self.d_ff, n_heads, top_p, dropout, in_dim)
                for _ in range(n_blocks)
            ]
        )
        self.n_blocks = n_blocks

    def forward(self, x, masks, alpha=0.5, is_training=False):
        # x: [B, L, D]
        moe_loss = 0.0
        for block in self.blocks:
            x, loss = block(x, masks, alpha, is_training)
            moe_loss = moe_loss + loss
        moe_loss = moe_loss / self.n_blocks
        return x, moe_loss


def _build_region_masks(L, N):
    """Build the S / T / ST partition masks for the patch graph.

    For node k (k in [0, L)) with N patches per channel:
      - S  : same intra-channel patch index, different node (spatial / cross-channel)
      - T  : same channel block (temporal, within the channel's patch window)
      - ST : everything else
    Returns a [L, 3, L] float tensor.
    """
    idx = torch.arange(L)
    masks = []
    for k in range(L):
        S = ((idx % N == k % N) & (idx != k)).float()
        T = ((idx >= (k // N) * N) & (idx < (k // N) * N + N)).float()
        ST = torch.ones(L) - S - T
        masks.append(torch.stack([S, T, ST], dim=0))
    return torch.stack(masks, dim=0)  # [L, 3, L]


class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        features="M",
        d_model=128,
        d_ff=256,
        n_heads=4,
        e_layers=2,
        patch_len=16,
        dropout=0.1,
        alpha=0.1,
        top_p=0.5,
        pos=True,
        c_out=None,
        **kwargs,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features
        self.n_vars = enc_in if c_out is None else c_out
        self.dim = d_model
        self.d_ff = d_ff
        self.patch_len = patch_len
        self.stride = patch_len
        self.num_patches = int((seq_len - patch_len) / self.stride + 1)  # per channel

        self.alpha = alpha
        self.top_p = top_p

        self.last_moe_loss = None

        # total number of patch-graph nodes = patches per channel * channels
        L = self.num_patches * self.n_vars
        # in_dim feeds the MoE gate (it sees rows of the [L, L] adjacency)
        in_dim = seq_len * self.n_vars // patch_len

        self.patch_embed = PatchEmbed(self.dim, self.patch_len, self.stride, pos)
        self.backbone = TimeFilter_Backbone(
            self.dim,
            self.n_vars,
            self.d_ff,
            n_heads,
            e_layers,
            self.top_p,
            dropout,
            in_dim,
        )
        self.head = nn.Linear(self.dim * self.num_patches, self.pred_len)

        self.use_RevIN = False
        self.norm = Normalize(enc_in, affine=self.use_RevIN)

        # precompute region masks [L, 3, L]
        self.register_buffer(
            "region_masks", _build_region_masks(L, self.num_patches), persistent=False
        )

    def forecast(self, x):
        # x: [B, T, C]
        B, T, C = x.shape
        x = self.norm(x, "norm")
        x = x.permute(0, 2, 1).reshape(-1, C * T)  # [B, C*T]
        x = self.patch_embed(x)  # [B, L, D]

        x, moe_loss = self.backbone(
            x, self.region_masks, self.alpha, self.training
        )
        self.last_moe_loss = moe_loss

        x = self.head(
            x.reshape(-1, self.n_vars, self.num_patches, self.dim).flatten(start_dim=-2)
        )  # [B, C, pred_len]
        x = x.permute(0, 2, 1)  # [B, pred_len, C]
        x = self.norm(x, "denorm")
        return x

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len :, :]  # [B, pred_len, C]
