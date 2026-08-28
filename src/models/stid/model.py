"""Licensed STID port pinned to BasicTS c218c07b6ce5e4cf908b147fd180c486346fed9c.

The backbone preserves BasicTS' Conv2d parameterization, residual MLP dropout,
initialization order, and four-dimensional history contract. The outer wrapper
only converts ModernTSF values and calendar marks to the BasicTS layout.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from components.marks import to_spatiotemporal


class _MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(input_dim, hidden_dim, kernel_size=(1, 1))
        self.fc2 = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=(1, 1))
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(input_data)))) + input_data


class _STID(nn.Module):
    """BasicTS STID architecture with explicit constructor arguments."""

    def __init__(
        self,
        input_len: int,
        output_len: int,
        num_features: int,
        input_dim: int = 3,
        input_hidden_size: int = 32,
        num_layers: int = 1,
        if_spatial: bool = True,
        spatial_hidden_size: int = 32,
        if_time_in_day: bool = True,
        if_day_in_week: bool = True,
        num_time_in_day: int = 24,
        num_day_in_week: int = 7,
        tid_hidden_size: int = 32,
        diw_hidden_size: int = 32,
    ) -> None:
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.input_dim = input_dim
        self.if_spatial = if_spatial
        self.if_time_in_day = if_time_in_day
        self.if_day_in_week = if_day_in_week
        self.num_time_in_day = num_time_in_day
        self.num_day_in_week = num_day_in_week

        if if_spatial:
            self.node_emb = nn.Parameter(torch.empty(num_features, spatial_hidden_size))
            nn.init.xavier_uniform_(self.node_emb)
        if if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(num_time_in_day, tid_hidden_size)
            )
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(num_day_in_week, diw_hidden_size)
            )
            nn.init.xavier_uniform_(self.day_in_week_emb)

        self.time_series_emb_layer = nn.Conv2d(
            input_dim * input_len, input_hidden_size, kernel_size=(1, 1)
        )
        hidden_size = (
            input_hidden_size
            + spatial_hidden_size * int(if_spatial)
            + tid_hidden_size * int(if_time_in_day)
            + diw_hidden_size * int(if_day_in_week)
        )
        self.encoder = nn.Sequential(
            *[_MultiLayerPerceptron(hidden_size, hidden_size) for _ in range(num_layers)]
        )
        self.regression_layer = nn.Conv2d(
            hidden_size, output_len, kernel_size=(1, 1)
        )

    def forward(
        self,
        history_data: torch.Tensor,
        future_data: torch.Tensor | None = None,
        batch_seen: int = 0,
        epoch: int = 0,
        train: bool = False,
    ) -> torch.Tensor:
        del future_data, batch_seen, epoch, train
        input_data = history_data[..., : self.input_dim]
        time_in_day_emb = (
            self.time_in_day_emb[
                (history_data[:, -1, :, 1] * self.num_time_in_day).long()
            ]
            if self.if_time_in_day
            else None
        )
        day_in_week_emb = (
            self.day_in_week_emb[
                (history_data[:, -1, :, 2] * self.num_day_in_week).long()
            ]
            if self.if_day_in_week
            else None
        )

        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = (
            input_data.view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        )
        embeddings = [self.time_series_emb_layer(input_data)]
        if self.if_spatial:
            embeddings.append(
                self.node_emb.unsqueeze(0)
                .expand(batch_size, -1, -1)
                .transpose(1, 2)
                .unsqueeze(-1)
            )
        if time_in_day_emb is not None:
            embeddings.append(time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            embeddings.append(day_in_week_emb.transpose(1, 2).unsqueeze(-1))
        return self.regression_layer(self.encoder(torch.cat(embeddings, dim=1)))


class Model(nn.Module):
    """ModernTSF input adapter around the pinned BasicTS STID backbone."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_nodes: int,
        adj_mx=None,
        input_dim: int = 3,
        embed_dim: int = 32,
        num_layers: int = 1,
        num_time_in_day: int = 24,
        num_day_in_week: int = 7,
        if_time_in_day: bool = True,
        if_day_in_week: bool = True,
    ) -> None:
        super().__init__()
        del adj_mx
        self.num_nodes = num_nodes
        self.net = _STID(
            input_len=seq_len,
            output_len=pred_len,
            num_features=num_nodes,
            input_dim=input_dim,
            input_hidden_size=embed_dim,
            num_layers=num_layers,
            if_spatial=True,
            spatial_hidden_size=embed_dim,
            if_time_in_day=if_time_in_day,
            if_day_in_week=if_day_in_week,
            num_time_in_day=num_time_in_day,
            num_day_in_week=num_day_in_week,
            tid_hidden_size=embed_dim,
            diw_hidden_size=embed_dim,
        )

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor | None = None,
        x_dec: torch.Tensor | None = None,
        x_mark_dec: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del x_dec, x_mark_dec, mask
        history = to_spatiotemporal(x_enc, x_mark_enc)
        return self.net(history, None, 0, 0, self.training)[..., 0]
