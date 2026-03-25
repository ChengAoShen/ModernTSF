"""CycleNet model implementation."""

from __future__ import annotations

import torch
import torch.nn as nn


class RecurrentCycle(nn.Module):
    def __init__(self, cycle_len: int, channel_size: int):
        super().__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = nn.Parameter(
            torch.zeros(cycle_len, channel_size), requires_grad=True
        )

    def forward(self, index: torch.Tensor, length: int) -> torch.Tensor:
        gather_index = (
            index.view(-1, 1) + torch.arange(length, device=index.device).view(1, -1)
        ) % self.cycle_len
        return self.data[gather_index]


class CycleNetModel(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        cycle_len: int,
        model_type: str,
        d_model: int,
        use_revin: bool,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.cycle_len = cycle_len
        self.model_type = model_type
        self.d_model = d_model
        self.use_revin = use_revin

        self.cycle_queue = RecurrentCycle(
            cycle_len=self.cycle_len, channel_size=self.enc_in
        )

        if self.model_type not in ["linear", "mlp"]:
            raise ValueError("model_type must be 'linear' or 'mlp'")
        if self.model_type == "linear":
            self.model = nn.Linear(self.seq_len, self.pred_len)
        else:
            self.model = nn.Sequential(
                nn.Linear(self.seq_len, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.pred_len),
            )

    def forward(self, x: torch.Tensor, cycle_index: torch.Tensor) -> torch.Tensor:
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        x = x - self.cycle_queue(cycle_index, self.seq_len)
        y = self.model(x.permute(0, 2, 1)).permute(0, 2, 1)
        y = y + self.cycle_queue(
            (cycle_index + self.seq_len) % self.cycle_len, self.pred_len
        )

        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean
        return y


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        cycle: int,
        model_type: str,
        d_model: int,
        use_revin: bool,
    ):
        super().__init__()
        self.model = CycleNetModel(
            seq_len=seq_len,
            pred_len=pred_len,
            enc_in=enc_in,
            cycle_len=cycle,
            model_type=model_type,
            d_model=d_model,
            use_revin=use_revin,
        )
        self.cycle = cycle

    def forward(self, x, x_time_stamp, *args):
        if self.cycle == 24:
            cycle_index = x_time_stamp[:, 0, 3].to(torch.int64)
        elif self.cycle == 7:
            cycle_index = x_time_stamp[:, 0, 4].to(torch.int64)
        elif self.cycle == 168:
            cycle_index = (x_time_stamp[:, 0, 4] * 24 + x_time_stamp[:, 0, 3]).to(
                torch.int64
            )
        else:
            cycle_index = x_time_stamp[:, 0, 3].to(torch.int64) % self.cycle
        return self.model(x, cycle_index)
