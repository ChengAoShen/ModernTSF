"""Verbatim STID model source.

Vendored from CauAir (src/models/stid.py).
BaseModel replaced with nn.Module; explicit params stored on self.
Reference: https://github.com/zezhishao/STID
"""

import torch
import torch.nn as nn


class STID(nn.Module):
    """Spatial-Temporal Identity model."""

    def __init__(self, tod, node_num, input_dim, output_dim,
                 seq_len, horizon):
        super(STID, self).__init__()
        self.node_num = node_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.horizon = horizon

        self.num_nodes = self.node_num
        self.node_dim = 32
        self.input_len = self.seq_len
        self.embed_dim = 32
        self.output_len = self.horizon
        self.num_layer = 3
        self.temp_dim_tid = 32
        self.temp_dim_diw = 32
        self.time_of_day_size = tod
        self.day_of_week_size = 7

        self.if_time_in_day = 1
        self.if_day_in_week = 1
        self.if_spatial = 1

        # spatial embeddings
        if self.if_spatial:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.node_dim))
            nn.init.xavier_uniform_(self.node_emb)
        if self.if_time_in_day:
            self.time_in_day_emb = nn.Parameter(
                torch.empty(self.time_of_day_size, self.temp_dim_tid))
            nn.init.xavier_uniform_(self.time_in_day_emb)
        if self.if_day_in_week:
            self.day_in_week_emb = nn.Parameter(
                torch.empty(self.day_of_week_size, self.temp_dim_diw))
            nn.init.xavier_uniform_(self.day_in_week_emb)

        # embedding layer
        self.time_series_emb_layer = nn.Conv2d(
            in_channels=self.input_dim * self.input_len,
            out_channels=self.embed_dim,
            kernel_size=(1, 1), bias=True)

        # encoding
        self.hidden_dim = (
            self.embed_dim
            + self.node_dim * int(self.if_spatial)
            + self.temp_dim_tid * int(self.if_day_in_week)
            + self.temp_dim_diw * int(self.if_time_in_day))
        self.encoder = nn.Sequential(
            *[STIDMultiLayerPerceptron(self.hidden_dim, self.hidden_dim)
              for _ in range(self.num_layer)])

        # regression
        self.regression_layer = nn.Conv2d(
            in_channels=self.hidden_dim,
            out_channels=self.output_len,
            kernel_size=(1, 1), bias=True)

    def forward(self, history_data, label=None, adj=None):
        input_data = history_data[..., range(self.input_dim)]

        if self.if_time_in_day:
            t_i_d_data = history_data[..., -2]
            time_in_day_emb = self.time_in_day_emb[
                (t_i_d_data[:, -1, :] * self.time_of_day_size
                 ).type(torch.LongTensor)]
        else:
            time_in_day_emb = None
        if self.if_day_in_week:
            d_i_w_data = history_data[..., -1]
            day_in_week_emb = self.day_in_week_emb[
                (d_i_w_data[:, -1, :] * self.day_of_week_size
                 ).type(torch.LongTensor)]
        else:
            day_in_week_emb = None

        # time series embedding
        batch_size, _, num_nodes, _ = input_data.shape
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(
            batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        time_series_emb = self.time_series_emb_layer(input_data)

        node_emb = []
        if self.if_spatial:
            node_emb.append(self.node_emb.unsqueeze(0).expand(
                batch_size, -1, -1).transpose(1, 2).unsqueeze(-1))
        tem_emb = []
        if time_in_day_emb is not None:
            tem_emb.append(
                time_in_day_emb.transpose(1, 2).unsqueeze(-1))
        if day_in_week_emb is not None:
            tem_emb.append(
                day_in_week_emb.transpose(1, 2).unsqueeze(-1))

        hidden = torch.cat(
            [time_series_emb] + node_emb + tem_emb, dim=1)
        hidden = self.encoder(hidden)
        prediction = self.regression_layer(hidden)
        return prediction

class STIDMultiLayerPerceptron(nn.Module):
    """Multi-Layer Perceptron with residual links."""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Conv2d(
            in_channels=input_dim, out_channels=hidden_dim,
            kernel_size=(1, 1), bias=True)
        self.fc2 = nn.Conv2d(
            in_channels=hidden_dim, out_channels=hidden_dim,
            kernel_size=(1, 1), bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.15)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))
        hidden = hidden + input_data
        return hidden
