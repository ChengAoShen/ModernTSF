"""xPatch building blocks."""

from __future__ import annotations

import torch
from torch import nn


class EMA(nn.Module):
    """Exponential moving average block."""

    def __init__(self, alpha: float) -> None:
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, t, _ = x.shape
        powers = torch.flip(
            torch.arange(t, device=x.device, dtype=torch.float64), dims=(0,)
        )
        alpha = torch.as_tensor(self.alpha, device=x.device, dtype=torch.float64)
        weights = torch.pow((1 - alpha), powers)
        divisor = weights.clone()
        weights[1:] = weights[1:] * alpha
        weights = weights.reshape(1, t, 1)
        divisor = divisor.reshape(1, t, 1)
        out = torch.cumsum(x.to(torch.float64) * weights, dim=1)
        out = torch.div(out, divisor)
        return out.to(dtype=x.dtype)


class DEMA(nn.Module):
    """Double exponential moving average block."""

    def __init__(self, alpha: float, beta: float) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = torch.as_tensor(self.alpha, device=x.device, dtype=x.dtype)
        beta = torch.as_tensor(self.beta, device=x.device, dtype=x.dtype)
        s_prev = x[:, 0, :]
        b = x[:, 1, :] - s_prev
        res = [s_prev.unsqueeze(1)]
        for t in range(1, x.shape[1]):
            xt = x[:, t, :]
            s = alpha * xt + (1 - alpha) * (s_prev + b)
            b = beta * (s - s_prev) + (1 - beta) * b
            s_prev = s
            res.append(s.unsqueeze(1))
        return torch.cat(res, dim=1)


class Decomp(nn.Module):
    """Series decomposition block."""

    def __init__(self, ma_type: str, alpha: float, beta: float) -> None:
        super().__init__()
        if ma_type == "ema":
            self.ma = EMA(alpha)
        elif ma_type == "dema":
            self.ma = DEMA(alpha, beta)
        else:
            raise ValueError(f"Unsupported ma_type: {ma_type}")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        moving_average = self.ma(x)
        res = x - moving_average
        return res, moving_average


class Network(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        patch_len: int,
        stride: int,
        padding_patch: str,
    ) -> None:
        super().__init__()
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.dim = patch_len * patch_len
        self.patch_num = (seq_len - patch_len) // stride + 1
        if padding_patch == "end":
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride))
            self.patch_num += 1

        self.fc1 = nn.Linear(patch_len, self.dim)
        self.gelu1 = nn.GELU()
        self.bn1 = nn.BatchNorm1d(self.patch_num)

        self.conv1 = nn.Conv1d(
            self.patch_num, self.patch_num, patch_len, patch_len, groups=self.patch_num
        )
        self.gelu2 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(self.patch_num)

        self.fc2 = nn.Linear(self.dim, patch_len)

        self.conv2 = nn.Conv1d(self.patch_num, self.patch_num, 1, 1)
        self.gelu3 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(self.patch_num)

        self.flatten1 = nn.Flatten(start_dim=-2)
        self.fc3 = nn.Linear(self.patch_num * patch_len, pred_len * 2)
        self.gelu4 = nn.GELU()
        self.fc4 = nn.Linear(pred_len * 2, pred_len)

        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)

        self.fc6 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)

        self.fc7 = nn.Linear(pred_len // 2, pred_len)

        self.fc8 = nn.Linear(pred_len * 2, pred_len)

    def forward(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        s = s.permute(0, 2, 1)
        t = t.permute(0, 2, 1)

        batch_size = s.shape[0]
        channels = s.shape[1]
        input_size = s.shape[2]
        s = torch.reshape(s, (batch_size * channels, input_size))
        t = torch.reshape(t, (batch_size * channels, input_size))

        if self.padding_patch == "end":
            s = self.padding_patch_layer(s)
        s = s.unfold(dimension=-1, size=self.patch_len, step=self.stride)

        s = self.fc1(s)
        s = self.gelu1(s)
        s = self.bn1(s)

        res = s

        s = self.conv1(s)
        s = self.gelu2(s)
        s = self.bn2(s)

        res = self.fc2(res)
        s = s + res

        s = self.conv2(s)
        s = self.gelu3(s)
        s = self.bn3(s)

        s = self.flatten1(s)
        s = self.fc3(s)
        s = self.gelu4(s)
        s = self.fc4(s)

        t = self.fc5(t)
        t = self.avgpool1(t)
        t = self.ln1(t)

        t = self.fc6(t)
        t = self.avgpool2(t)
        t = self.ln2(t)

        t = self.fc7(t)

        x = torch.cat((s, t), dim=1)
        x = self.fc8(x)

        x = torch.reshape(x, (batch_size, channels, self.pred_len))
        x = x.permute(0, 2, 1)
        return x
