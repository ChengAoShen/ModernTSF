"""Verbatim DeepAir model source.

Vendored from CauAir (src/models/deepair.py).
BaseModel replaced with nn.Module; explicit params stored on self.
"""

import torch
import torch.nn as nn


class DeepAir(nn.Module):
    """DeepAir fusion-based forecasting model."""

    def __init__(self, d_hid, node_num, input_dim, output_dim,
                 seq_len, horizon):
        super(DeepAir, self).__init__()
        self.node_num = node_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.horizon = horizon

        self.FusionNet = nn.ModuleList([
            FusionNet(d_hid * (self.input_dim if i == 0 else 2))
            for i in range(self.input_dim)])

        self.emb = nn.ModuleList([
            nn.Conv1d(self.seq_len, d_hid, kernel_size=1)
            for _ in range(self.input_dim)])

        self.merge = nn.ModuleList([
            nn.Conv1d(
                d_hid * (self.input_dim if i == 0 else 2),
                self.output_dim * self.horizon,
                kernel_size=1, bias=False)
            for i in range(self.input_dim)])

    def forward(self, x, label=None, adj=None):
        h_x = self.emb[0](x[..., 0])
        feature = [h_x]
        output = 0
        for i in range(1, self.input_dim):
            h = self.emb[i](x[..., i])
            feature.append(h)
            z = self.FusionNet[i](torch.cat([h_x, h], dim=1))
            output += self.merge[i](z)

        z = self.FusionNet[0](torch.cat(feature, dim=1))
        output += self.merge[0](z)
        return output.unsqueeze(-1)


class FusionNet(nn.Module):
    def __init__(self, d_hid):
        super().__init__()
        self.fc = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_hid, d_hid, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(d_hid, d_hid, kernel_size=1),
            )
            for _ in range(3)])

    def forward(self, x):
        x = self.fc[0](x)
        x = self.fc[1](x) + x
        x = self.fc[2](x)
        return x
