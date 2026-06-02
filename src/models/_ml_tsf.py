"""PyTorch-native machine-learning and statistical TSF adapters.

These modules expose classical forecasting families through the same
``torch.nn.Module`` contract as the rest of ModernTSF.  Tree/boosting,
kernel/prototype, ARIMA-like, smoothing, and recurrent baselines therefore use
the standard trainer and can run on CUDA/MPS when the benchmark device is set
accordingly.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.module.revin import RevIN


def _tree_routes(depth: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Return heap-indexed routes and left/right decisions for soft trees."""
    leaves = 2**depth
    routes = torch.zeros(leaves, depth, dtype=torch.long)
    directions = torch.zeros(leaves, depth, dtype=torch.float32)
    for leaf in range(leaves):
        node = 0
        for level in range(depth):
            bit = (leaf >> (depth - level - 1)) & 1
            routes[leaf, level] = node
            directions[leaf, level] = float(bit)
            node = 2 * node + 1 + bit
    return routes, directions


class SoftTreeEnsemble(nn.Module):
    """Differentiable tree ensemble over flattened lag features."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_trees: int = 16,
        depth: int = 3,
        temperature: float = 1.0,
        randomize: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_trees = max(1, num_trees)
        self.depth = max(1, depth)
        self.temperature = temperature
        splits = 2**self.depth - 1
        leaves = 2**self.depth

        self.split_weight = nn.Parameter(
            torch.empty(self.num_trees, splits, input_dim)
        )
        self.split_bias = nn.Parameter(torch.zeros(self.num_trees, splits))
        self.leaf_value = nn.Parameter(
            torch.empty(self.num_trees, leaves, output_dim)
        )
        routes, directions = _tree_routes(self.depth)
        self.register_buffer("routes", routes)
        self.register_buffer("directions", directions)

        scale = 0.5 if randomize else 0.1
        nn.init.normal_(self.split_weight, std=scale / math.sqrt(max(1, input_dim)))
        nn.init.normal_(self.leaf_value, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, input_dim)
        split_logits = torch.einsum("bi,tsi->bts", x, self.split_weight)
        split_logits = (split_logits + self.split_bias.unsqueeze(0)) * self.temperature
        split_prob = torch.sigmoid(split_logits)

        leaf_prob = x.new_ones(x.size(0), self.num_trees, self.routes.size(0))
        for level in range(self.depth):
            route_idx = self.routes[:, level]
            direction = self.directions[:, level].view(1, 1, -1)
            probs = split_prob[:, :, route_idx]
            leaf_prob = leaf_prob * torch.where(direction > 0.5, probs, 1.0 - probs)

        out = torch.einsum("btl,tlo->bo", leaf_prob, self.leaf_value)
        return out / float(self.num_trees)


class MLTSFModel(nn.Module):
    """Unified GPU-capable adapter for classical ML TSF baselines."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        family: str,
        variant: str,
        d_model: int = 64,
        dropout: float = 0.1,
        num_layers: int = 1,
        num_estimators: int = 16,
        tree_depth: int = 3,
        num_prototypes: int = 32,
        kernel_gamma: float = 0.1,
        l1_penalty: float = 0.0,
        l2_penalty: float = 0.0,
        use_revin: bool = True,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.family = family
        self.variant = variant
        self.kernel_gamma = kernel_gamma
        self.l1_penalty = l1_penalty
        self.l2_penalty = l2_penalty
        self.use_revin = use_revin
        self.aux_loss: torch.Tensor | None = None

        self.revin = RevIN(enc_in, affine=True, subtract_last=False)
        self.channel_mixer = nn.Sequential(
            nn.Linear(enc_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, enc_in),
        )

        flat_dim = seq_len * enc_in
        out_dim = pred_len * enc_in
        self.linear_head = nn.Linear(seq_len, pred_len)
        self.poly_head = nn.Linear(seq_len * 3, pred_len)
        self.mlp_head = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, pred_len),
        )
        self.flat_skip = nn.Linear(flat_dim, out_dim)

        diff_len = max(1, seq_len - 1)
        self.diff_head = nn.Linear(diff_len, pred_len)
        self.decay_logits = nn.Parameter(torch.linspace(-1.0, 1.0, seq_len))
        self.trend_scale = nn.Parameter(torch.tensor(1.0))
        self.kalman_alpha = nn.Parameter(torch.tensor(0.0))
        self.kalman_beta = nn.Parameter(torch.tensor(-1.0))

        self.prototypes = nn.Parameter(torch.empty(num_prototypes, flat_dim))
        self.prototype_values = nn.Parameter(torch.empty(num_prototypes, out_dim))
        nn.init.normal_(self.prototypes, std=0.02)
        nn.init.normal_(self.prototype_values, std=0.02)

        effective_estimators = 1 if family == "decision_tree" else num_estimators
        effective_depth = 2 if family == "extra_trees" else tree_depth
        randomize = family in {"extra_trees", "random_forest"}
        temperature = 1.5 if family in {"xgboost", "lightgbm", "catboost"} else 1.0
        self.trees = SoftTreeEnsemble(
            input_dim=flat_dim,
            output_dim=out_dim,
            num_trees=effective_estimators,
            depth=effective_depth,
            temperature=temperature,
            randomize=randomize,
        )
        self.boost_scale = nn.Parameter(torch.tensor(1.0))

        recurrent_cls: type[nn.RNNBase] | None
        recurrent_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}.get(family)
        self.recurrent = None
        if recurrent_cls is not None:
            self.recurrent = recurrent_cls(
                input_size=enc_in,
                hidden_size=d_model,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.recurrent_head = nn.Linear(d_model, out_dim)
        else:
            self.recurrent_head = nn.Linear(d_model, out_dim)

        self.tcn = nn.Sequential(
            nn.Conv1d(enc_in, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
        )
        self.tcn_head = nn.Linear(d_model, out_dim)

    def _regularization(self) -> torch.Tensor:
        reg = self.linear_head.weight.new_tensor(0.0)
        if self.l1_penalty:
            reg = reg + self.l1_penalty * sum(
                p.abs().mean() for p in self.parameters() if p.requires_grad
            )
        if self.l2_penalty:
            reg = reg + self.l2_penalty * sum(
                p.square().mean() for p in self.parameters() if p.requires_grad
            )
        return reg

    def _linear_family(self, xc: torch.Tensor) -> torch.Tensor:
        # xc: (B, C, L)
        if self.family == "polynomial":
            features = torch.cat([xc, xc.square(), torch.sqrt(xc.abs() + 1e-6)], dim=-1)
            out = self.poly_head(features)
        elif self.family == "mlp":
            out = self.mlp_head(xc)
        else:
            out = self.linear_head(xc)
        return out.transpose(1, 2)

    def _statistical_family(self, xc: torch.Tensor) -> torch.Tensor:
        if self.family == "arima":
            if self.seq_len > 1:
                diff = xc[..., 1:] - xc[..., :-1]
            else:
                diff = xc.new_zeros(xc.size(0), xc.size(1), 1)
            delta = self.diff_head(diff)
            return (xc[..., -1:] + torch.cumsum(delta, dim=-1)).transpose(1, 2)

        if self.family == "autoreg":
            return self.linear_head(xc).transpose(1, 2)

        if self.family == "exp_smoothing":
            weights = torch.softmax(self.decay_logits, dim=0)
            level = torch.sum(xc * weights.view(1, 1, -1), dim=-1)
            denom = max(1, self.seq_len - 1)
            trend = (xc[..., -1] - xc[..., 0]) / float(denom)
            horizon = torch.arange(
                1, self.pred_len + 1, device=xc.device, dtype=xc.dtype
            )
            out = level.unsqueeze(-1) + self.trend_scale * trend.unsqueeze(-1) * horizon
            return out.transpose(1, 2)

        # Alpha-beta Kalman-style smoother.
        alpha = torch.sigmoid(self.kalman_alpha)
        beta = torch.sigmoid(self.kalman_beta)
        level = xc[..., 0]
        trend = torch.zeros_like(level)
        for idx in range(1, self.seq_len):
            prediction = level + trend
            error = xc[..., idx] - prediction
            level = prediction + alpha * error
            trend = trend + beta * error
        horizon = torch.arange(
            1, self.pred_len + 1, device=xc.device, dtype=xc.dtype
        )
        out = level.unsqueeze(-1) + trend.unsqueeze(-1) * horizon
        return out.transpose(1, 2)

    def _prototype_family(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(1)
        distances = torch.cdist(flat, self.prototypes).square()
        gamma = torch.as_tensor(self.kernel_gamma, device=x.device, dtype=x.dtype)
        weights = torch.softmax(-gamma.clamp_min(1e-6) * distances, dim=-1)
        out = weights @ self.prototype_values
        if self.family == "svr":
            out = out + 0.5 * self.flat_skip(flat)
        if self.family == "gaussian_process":
            out = out + 0.25 * self.flat_skip(flat)
        return out.view(x.size(0), self.pred_len, self.enc_in)

    def _tree_family(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.flatten(1)
        tree_out = self.trees(flat)
        if self.family in {"xgboost", "lightgbm", "catboost", "gradient_boosting"}:
            tree_out = self.flat_skip(flat) + self.boost_scale * tree_out
        return tree_out.view(x.size(0), self.pred_len, self.enc_in)

    def _recurrent_family(self, x: torch.Tensor) -> torch.Tensor:
        if self.family == "tcn":
            hidden = self.tcn(x.transpose(1, 2)).mean(dim=-1)
            return self.tcn_head(hidden).view(x.size(0), self.pred_len, self.enc_in)
        assert self.recurrent is not None
        output, state = self.recurrent(x)
        if isinstance(state, tuple):
            hidden = state[0][-1]
        else:
            hidden = state[-1]
        hidden = hidden + output[:, -1, :]
        return self.recurrent_head(hidden).view(x.size(0), self.pred_len, self.enc_in)

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        if self.use_revin:
            x = self.revin(x, "norm")

        xc = x.transpose(1, 2)
        if self.family in {"linear", "ridge", "lasso", "elastic_net", "polynomial", "mlp"}:
            out = self._linear_family(xc)
        elif self.family in {"arima", "autoreg", "exp_smoothing", "kalman"}:
            out = self._statistical_family(xc)
        elif self.family in {"knn", "svr", "gaussian_process"}:
            out = self._prototype_family(x)
        elif self.family in {
            "decision_tree",
            "random_forest",
            "extra_trees",
            "gradient_boosting",
            "xgboost",
            "lightgbm",
            "catboost",
        }:
            out = self._tree_family(x)
        elif self.family in {"rnn", "gru", "lstm", "tcn"}:
            out = self._recurrent_family(x)
        else:
            raise ValueError(f"unknown MLTSF family: {self.family}")

        out = out + 0.1 * self.channel_mixer(out)
        self.aux_loss = self._regularization()
        if self.use_revin:
            out = self.revin(out, "denorm")
        return out
