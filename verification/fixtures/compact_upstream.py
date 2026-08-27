"""Thin executable mirrors for pinned FITS, SparseTSF, and CycleNet sources.

These Apache-2.0 fixtures keep ordinary test runs independent of network
checkouts. Evidence generation still imports and runs each exact pinned
upstream file; file hashes prevent this mirror from substituting for that run.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FITS(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = configs.individual
        self.channels = configs.enc_in
        self.dominance_freq = configs.cut_freq
        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len
        if self.individual:
            self.freq_upsampler = nn.ModuleList(
                nn.Linear(
                    self.dominance_freq,
                    int(self.dominance_freq * self.length_ratio),
                ).to(torch.cfloat)
                for _ in range(self.channels)
            )
        else:
            self.freq_upsampler = nn.Linear(
                self.dominance_freq,
                int(self.dominance_freq * self.length_ratio),
            ).to(torch.cfloat)

    def forward(self, x):
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        x = x / torch.sqrt(x_var)
        low_specx = torch.fft.rfft(x, dim=1)
        low_specx[:, self.dominance_freq :] = 0
        low_specx = low_specx[:, : self.dominance_freq, :]
        if self.individual:
            low_specxy_ = torch.zeros(
                x.size(0),
                int(self.dominance_freq * self.length_ratio),
                x.size(2),
                dtype=low_specx.dtype,
                device=x.device,
            )
            for index in range(self.channels):
                low_specxy_[:, :, index] = self.freq_upsampler[index](
                    low_specx[:, :, index]
                )
        else:
            low_specxy_ = self.freq_upsampler(low_specx.permute(0, 2, 1)).permute(
                0, 2, 1
            )
        low_specxy = torch.zeros(
            x.size(0),
            int((self.seq_len + self.pred_len) / 2 + 1),
            x.size(2),
            dtype=low_specxy_.dtype,
            device=x.device,
        )
        low_specxy[:, : low_specxy_.size(1), :] = low_specxy_
        low_xy = torch.fft.irfft(low_specxy, dim=1) * self.length_ratio
        return low_xy * torch.sqrt(x_var) + x_mean, low_xy * torch.sqrt(x_var)


class SparseTSF(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len
        self.d_model = configs.d_model
        self.model_type = configs.model_type
        self.seg_num_x = self.seq_len // self.period_len
        self.seg_num_y = self.pred_len // self.period_len
        self.conv1d = nn.Conv1d(
            1,
            1,
            kernel_size=1 + 2 * (self.period_len // 2),
            padding=self.period_len // 2,
            bias=False,
        )
        if self.model_type == "linear":
            self.linear = nn.Linear(self.seg_num_x, self.seg_num_y, bias=False)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(self.seg_num_x, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.seg_num_y),
            )

    def forward(self, x):
        batch_size = x.shape[0]
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = (x - seq_mean).permute(0, 2, 1)
        x = (
            self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(
                -1, self.enc_in, self.seq_len
            )
            + x
        )
        x = x.reshape(-1, self.seg_num_x, self.period_len).permute(0, 2, 1)
        y = self.linear(x) if self.model_type == "linear" else self.mlp(x)
        y = y.permute(0, 2, 1).reshape(batch_size, self.enc_in, self.pred_len)
        return y.permute(0, 2, 1) + seq_mean


class RecurrentCycle(nn.Module):
    def __init__(self, cycle_len, channel_size):
        super().__init__()
        self.cycle_len = cycle_len
        self.data = nn.Parameter(torch.zeros(cycle_len, channel_size))

    def forward(self, index, length):
        gather_index = (
            index.view(-1, 1)
            + torch.arange(length, device=index.device).view(1, -1)
        ) % self.cycle_len
        return self.data[gather_index]


class CycleNet(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.cycle_len = configs.cycle
        self.model_type = configs.model_type
        self.d_model = configs.d_model
        self.use_revin = configs.use_revin
        self.cycleQueue = RecurrentCycle(self.cycle_len, self.enc_in)
        if self.model_type == "linear":
            self.model = nn.Linear(self.seq_len, self.pred_len)
        else:
            self.model = nn.Sequential(
                nn.Linear(self.seq_len, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.pred_len),
            )

    def forward(self, x, cycle_index):
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)
        x = x - self.cycleQueue(cycle_index, self.seq_len)
        y = self.model(x.permute(0, 2, 1)).permute(0, 2, 1)
        y = y + self.cycleQueue(
            (cycle_index + self.seq_len) % self.cycle_len,
            self.pred_len,
        )
        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean
        return y
