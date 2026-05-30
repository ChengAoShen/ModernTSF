"""Core BigST spatio-temporal model (single-stage path).

Vendored/adapted from https://github.com/GestaltCogTeam/BasicTS
(baselines/BigST/arch/model.py), Apache-2.0.

Only the single-stage ``use_long=False`` path is vendored: the two-stage
``use_long=True`` path loads a pre-trained ``BigSTPreprocess`` checkpoint from
disk (a separate, dataset-scale preprocessing run) which cannot run on a tiny
smoke bundle. The model body is otherwise unchanged except:

* ``x[..., 1]`` / ``x[..., 2]`` are cast with ``.long()`` (device-preserving)
  instead of ``.type(torch.LongTensor)`` (which forced CPU), and the
  day-of-week channel is scaled by ``day_of_week_size`` before indexing so the
  ModernTSF ``[0, 1)``-normalized calendar covariates land on valid rows.
* ``torch.qr`` -> ``torch.linalg.qr`` (in the random-map helper).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.bigst._linear_conv import linearized_conv


class Model(nn.Module):
    def __init__(
        self,
        seq_num,
        in_dim,
        out_dim,
        hid_dim,
        num_nodes,
        tau,
        random_feature_dim,
        node_emb_dim,
        time_emb_dim,
        use_residual,
        use_bn,
        use_spatial,
        use_long,
        dropout,
        time_of_day_size,
        day_of_week_size,
        supports=None,
        edge_indices=None,
    ):
        super().__init__()

        self.tau = tau
        self.layer_num = 3
        self.in_dim = in_dim
        self.random_feature_dim = random_feature_dim

        self.use_residual = use_residual
        self.use_bn = use_bn
        self.use_spatial = use_spatial
        self.use_long = use_long

        self.dropout = dropout
        self.activation = nn.ReLU()
        self.supports = supports

        self.time_num = time_of_day_size
        self.week_num = day_of_week_size

        # node embedding layer
        self.node_emb_layer = nn.Parameter(torch.empty(num_nodes, node_emb_dim))
        nn.init.xavier_uniform_(self.node_emb_layer)

        # time embedding layer
        self.time_emb_layer = nn.Parameter(torch.empty(self.time_num, time_emb_dim))
        nn.init.xavier_uniform_(self.time_emb_layer)
        self.week_emb_layer = nn.Parameter(torch.empty(self.week_num, time_emb_dim))
        nn.init.xavier_uniform_(self.week_emb_layer)

        # embedding layer
        self.input_emb_layer = nn.Conv2d(seq_num * in_dim, hid_dim, kernel_size=(1, 1), bias=True)

        # ``x_g`` (the graph-feature stack) and the main feature stack ``x``
        # are concatenations of node + 2x time embeddings (and, for ``x``, the
        # input embedding). Upstream this equals ``hid_dim * 4`` because
        # ``node_emb_dim == time_emb_dim == hid_dim``; here we compute the true
        # widths so the embedding dims can differ from ``hid_dim``.
        graph_dim = node_emb_dim + time_emb_dim * 2
        feat_dim = hid_dim + node_emb_dim + time_emb_dim * 2
        self.feat_dim = feat_dim

        self.W_1 = nn.Conv2d(graph_dim, node_emb_dim, kernel_size=(1, 1), bias=True)
        self.W_2 = nn.Conv2d(graph_dim, node_emb_dim, kernel_size=(1, 1), bias=True)

        self.linear_conv = nn.ModuleList()
        self.bn = nn.ModuleList()

        self.supports_len = 0
        if supports is not None:
            self.supports_len += len(supports)

        for _ in range(self.layer_num):
            self.linear_conv.append(
                linearized_conv(feat_dim, feat_dim, self.dropout, self.tau, self.random_feature_dim)
            )
            self.bn.append(nn.LayerNorm(feat_dim))

        self.regression_layer = nn.Conv2d(feat_dim * 2, out_dim, kernel_size=(1, 1), bias=True)

    def forward(self, x, feat=None):
        # x: (B, N, T, D)
        B, N, T, D = x.size()

        tod_idx = (x[:, :, -1, 1] * self.time_num).long().clamp_(0, self.time_num - 1)
        dow_idx = (x[:, :, -1, 2] * self.week_num).long().clamp_(0, self.week_num - 1)
        time_emb = self.time_emb_layer[tod_idx]
        week_emb = self.week_emb_layer[dow_idx]

        # input embedding
        x = x.contiguous().view(B, N, -1).transpose(1, 2).unsqueeze(-1)  # (B, D*T, N, 1)
        input_emb = self.input_emb_layer(x)

        # node embeddings
        node_emb = self.node_emb_layer.unsqueeze(0).expand(B, -1, -1).transpose(1, 2).unsqueeze(-1)

        # time embeddings
        time_emb = time_emb.transpose(1, 2).unsqueeze(-1)  # (B, dim, N, 1)
        week_emb = week_emb.transpose(1, 2).unsqueeze(-1)  # (B, dim, N, 1)

        x_g = torch.cat([node_emb, time_emb, week_emb], dim=1)  # (B, dim*4, N, 1)
        x = torch.cat([input_emb, node_emb, time_emb, week_emb], dim=1)  # (B, dim*4, N, 1)

        # linearized spatial convolution
        x_pool = [x]  # (B, dim*4, N, 1)
        node_vec1 = self.W_1(x_g)  # (B, dim, N, 1)
        node_vec2 = self.W_2(x_g)  # (B, dim, N, 1)
        node_vec1 = node_vec1.permute(0, 2, 3, 1)  # (B, N, 1, dim)
        node_vec2 = node_vec2.permute(0, 2, 3, 1)  # (B, N, 1, dim)
        node_vec1_prime = node_vec2_prime = None
        for i in range(self.layer_num):
            if self.use_residual:
                residual = x
            x, node_vec1_prime, node_vec2_prime = self.linear_conv[i](x, node_vec1, node_vec2)

            if self.use_residual:
                x = x + residual

            if self.use_bn:
                x = x.permute(0, 2, 3, 1)  # (B, N, 1, dim*4)
                x = self.bn[i](x)
                x = x.permute(0, 3, 1, 2)

        x_pool.append(x)
        x = torch.cat(x_pool, dim=1)  # (B, dim*4, N, 1)

        x = self.activation(x)  # (B, dim*4, N, 1)

        x = self.regression_layer(x)  # (B, out_dim, N, 1)
        x = x.squeeze(-1).permute(0, 2, 1)  # (B, N, out_dim)

        return {
            "prediction": x.transpose(1, 2).unsqueeze(-1),  # (B, out_dim, N, 1)
            "node_vec1": node_vec1_prime,
            "node_vec2": node_vec2_prime,
            "supports": self.supports,
            "use_spatial": self.use_spatial,
        }
