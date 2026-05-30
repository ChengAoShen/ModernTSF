"""Linearized spatial convolution for BigST.

Vendored/adapted from https://github.com/GestaltCogTeam/BasicTS
(baselines/BigST/arch/linear_conv.py), Apache-2.0. The unused (commented-out)
``spatial_loss`` helper and the ``pdb`` import were dropped; behaviour is
unchanged. All device handling derives from the input tensors.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from models.bigst._random_map import create_random_matrix, random_feature_map


def linear_kernel(x, node_vec1, node_vec2):
    # x: [B, N, 1, nhid] node_vec1: [B, N, 1, r], node_vec2: [B, N, 1, r]
    node_vec1 = node_vec1.permute(1, 0, 2, 3)  # [N, B, 1, r]
    node_vec2 = node_vec2.permute(1, 0, 2, 3)  # [N, B, 1, r]
    x = x.permute(1, 0, 2, 3)  # [N, B, 1, nhid]

    v2x = torch.einsum("nbhm,nbhd->bhmd", node_vec2, x)
    out1 = torch.einsum("nbhm,bhmd->nbhd", node_vec1, v2x)  # [N, B, 1, nhid]

    one_matrix = torch.ones([node_vec2.shape[0]]).to(node_vec1.device)
    node_vec2_sum = torch.einsum("nbhm,n->bhm", node_vec2, one_matrix)
    out2 = torch.einsum("nbhm,bhm->nbh", node_vec1, node_vec2_sum)  # [N, 1]

    out1 = out1.permute(1, 0, 2, 3)  # [B, N, 1, nhid]
    out2 = out2.permute(1, 0, 2)
    out2 = torch.unsqueeze(out2, len(out2.shape))
    out = out1 / out2  # [B, N, 1, nhid]

    return out


class conv_approximation(nn.Module):
    def __init__(self, dropout, tau, random_feature_dim):
        super().__init__()
        self.tau = tau
        self.random_feature_dim = random_feature_dim
        self.activation = nn.ReLU()
        self.dropout = dropout

    def forward(self, x, node_vec1, node_vec2):
        dim = node_vec1.shape[-1]  # (N, 1, d)

        random_seed = torch.ceil(torch.abs(torch.sum(node_vec1) * 1e8)).to(torch.int32)
        random_matrix = create_random_matrix(
            self.random_feature_dim, dim, seed=random_seed
        ).to(node_vec1.device)  # (d, r)

        node_vec1 = node_vec1 / math.sqrt(self.tau)
        node_vec2 = node_vec2 / math.sqrt(self.tau)
        node_vec1_prime = random_feature_map(node_vec1, True, random_matrix)  # [B, N, 1, r]
        node_vec2_prime = random_feature_map(node_vec2, False, random_matrix)  # [B, N, 1, r]

        x = linear_kernel(x, node_vec1_prime, node_vec2_prime)

        return x, node_vec1_prime, node_vec2_prime


class linearized_conv(nn.Module):
    def __init__(self, in_dim, hid_dim, dropout, tau=1.0, random_feature_dim=64):
        super().__init__()

        self.dropout = dropout
        self.tau = tau
        self.random_feature_dim = random_feature_dim

        self.input_fc = nn.Conv2d(
            in_channels=in_dim, out_channels=hid_dim, kernel_size=(1, 1), bias=True
        )
        self.activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(p=dropout)

        self.conv_app_layer = conv_approximation(self.dropout, self.tau, self.random_feature_dim)

    def forward(self, input_data, node_vec1, node_vec2):
        x = self.input_fc(input_data)
        x = self.activation(x)
        x = self.dropout_layer(x)

        x = x.permute(0, 2, 3, 1)  # (B, N, 1, dim*4)
        x, node_vec1_prime, node_vec2_prime = self.conv_app_layer(x, node_vec1, node_vec2)
        x = x.permute(0, 3, 1, 2)  # (B, dim*4, N, 1)

        return x, node_vec1_prime, node_vec2_prime
