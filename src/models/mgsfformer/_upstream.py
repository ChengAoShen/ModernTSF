"""Verbatim MGSFformer model source.

Vendored from CauAir (src/models/mgsfformer.py).
BaseModel replaced with nn.Module; explicit dimension params added.
"""

import torch
from torch import nn
import torch.nn.functional as F


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True,
                 subtract_last=False):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(
                x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True,
                      unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x


    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class IE_block(nn.Module):
    def __init__(self, input_num, out_num, IE_Input_len):
        super(IE_block, self).__init__()
        self.IE_Input_len = IE_Input_len
        self.output = nn.Linear(input_num, out_num)

    def forward(self, x):
        x = x.reshape((x.shape[0], x.shape[1], x.shape[2], 1))
        x = IE_block.piecewise_sample(x, self.IE_Input_len)
        x = self.output(x)
        return x

    @staticmethod
    def piecewise_sample(data, n):
        result = 0.0
        data_len = data.shape[2] // n
        for i in range(n):
            line = data[:, :, data_len * i:data_len * (i + 1), :]
            if i == 0:
                result = line
            else:
                result = torch.cat([result, line], dim=3)
        result = result.transpose(2, 3)
        return result


class Time_att(nn.Module):
    """Temporal attention."""

    def __init__(self, Input_len, dim_input, dropout, num_head):
        super(Time_att, self).__init__()
        self.query = nn.Conv2d(
            in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.key = nn.Conv2d(
            in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.value = nn.Conv2d(
            in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.laynorm = nn.LayerNorm([Input_len])
        self.softmax = nn.Softmax(dim=-1)
        self.num_head = num_head
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(num_head, 1)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        result = 0.0
        for i in range(self.num_head):
            q = self.dropout(self.query(x)).transpose(-3, -2)
            k = self.dropout(self.key(x)).permute(0, 2, 3, 1)
            v = self.dropout(self.value(x)).transpose(-3, -2)
            kd = torch.sqrt(
                torch.tensor(k.shape[-1]).to(torch.float32)
                / self.num_head)
            line = self.dropout(self.softmax(q @ k / kd)) @ v
            if i < 1:
                result = line.unsqueeze(-1)
            else:
                result = torch.cat(
                    [result, line.unsqueeze(-1)], dim=-1)
        result = self.output(result)
        result = result.squeeze(-1)
        x = x + result.transpose(-3, -2)
        x = self.laynorm(x)
        return x


class space_att2(nn.Module):
    """Space attention."""

    def __init__(self, dim_input, dropout, num_head):
        super(space_att2, self).__init__()
        self.query = nn.Linear(dim_input, dim_input)
        self.key = nn.Linear(dim_input, dim_input)
        self.value = nn.Linear(dim_input, dim_input)
        self.softmax = nn.Softmax(dim=-1)
        self.num_head = num_head
        self.linear1 = nn.Linear(num_head, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        result = 0.0
        q = self.dropout(self.query(x))
        k = self.dropout(self.key(x))
        k = k.transpose(-2, -1)
        v = self.dropout(self.value(x))
        kd = torch.sqrt(
            torch.tensor(k.shape[-1]).to(torch.float32)
            / self.num_head)

        for i in range(self.num_head):
            line = self.dropout(self.softmax(q @ k / kd)) @ v
            if i < 1:
                result = line.unsqueeze(-1)
            else:
                result = torch.cat(
                    [result, line.unsqueeze(-1)], dim=-1)
        result = self.linear1(result)
        result = result.squeeze(-1)
        result = result.permute(0, 2, 3, 1)
        return result


class cross_att(nn.Module):
    """Cross attention."""

    def __init__(self, Input_len, dim_input, dropout, num_head):
        super(cross_att, self).__init__()
        self.query = nn.Conv2d(
            in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.key = nn.Conv2d(
            in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.value = nn.Conv2d(
            in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.laynorm = nn.LayerNorm([Input_len])
        self.softmax = nn.Softmax(dim=-1)
        self.num_head = num_head
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(num_head, 1)

    def forward(self, x, x2):
        result = 0.0
        for i in range(self.num_head):
            q = self.dropout(self.query(x2)).transpose(-3, -2)
            k = self.dropout(self.key(x)).transpose(-3, -2)
            k = k.transpose(-2, -1)
            v = self.dropout(self.value(x)).transpose(-3, -2)
            kd = torch.sqrt(
                torch.tensor(k.shape[-1]).to(torch.float32)
                / self.num_head)
            line = self.dropout(self.softmax(q @ k / kd)) @ v
            if i < 1:
                result = line.unsqueeze(-1)
            else:
                result = torch.cat(
                    [result, line.unsqueeze(-1)], dim=-1)
        result = self.output(result)
        result = result.squeeze(-1)
        x = x.transpose(-3, -2) + result
        x = self.laynorm(x)
        return x


class STA_block_att(nn.Module):
    def __init__(self, Input_len, num_id, IE_dim, out_len,
                 dropout, num_head):
        super(STA_block_att, self).__init__()
        self.Time_att = Time_att(Input_len, IE_dim, dropout, num_head)
        self.space_att = space_att2(num_id, dropout, num_head)
        self.cross_att = cross_att(Input_len, IE_dim, dropout, num_head)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Conv1d(
            in_channels=Input_len * IE_dim,
            out_channels=out_len, kernel_size=1)

    def forward(self, x):
        x = self.cross_att(self.Time_att(x), self.space_att(x))
        x = x.reshape((x.shape[0], x.shape[1], -1))
        x = self.linear(x.transpose(-2, -1))
        return x.transpose(-2, -1)


class RF_att(nn.Module):
    def __init__(self, dim_input):
        super(RF_att, self).__init__()
        self.QK = nn.Linear(dim_input, dim_input)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q_K = self.QK(x)
        Q_K = self.softmax(Q_K)
        x = x * Q_K
        return x


class DF_block(nn.Module):
    def __init__(self, num_ga, out_len):
        super(DF_block, self).__init__()
        self.att1 = RF_att(num_ga)
        self.att2 = RF_att(num_ga)
        self.att3 = RF_att(num_ga)
        self.out_len = out_len // 4

    def forward(self, x):
        line1 = x[:, :, 0:self.out_len, :]
        line1 = self.att1(line1)
        line1 = line1.sum(dim=-1)

        line2 = x[:, :, self.out_len:self.out_len * 2, :]
        line2 = self.att1(line2)
        line2 = line2.sum(dim=-1)

        line3 = x[:, :, self.out_len * 2:, :]
        line3 = self.att1(line3)
        line3 = line3.sum(dim=-1)
        x = torch.cat([line1, line2, line3], dim=2)

        return x


class MGSFformer(nn.Module):
    """MGSFformer multi-granularity spatiotemporal model."""

    def __init__(self,
                 node_num,
                 input_dim,
                 output_dim,
                 seq_len,
                 horizon,
                 Input_len=None,
                 out_len=None,
                 num_id=None,
                 IE_dim=32,
                 dropout=0.3,
                 num_head=2):
        super(MGSFformer, self).__init__()
        self.node_num = node_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = seq_len
        self.horizon = horizon

        # Allow explicit override or use base params
        Input_len = Input_len if Input_len is not None else seq_len
        out_len = out_len if out_len is not None else horizon
        num_id = num_id if num_id is not None else node_num

        self.RevIN = RevIN(num_id)

        # RD-block
        self.IE_Input_len = Input_len // 24
        self.IE_block1 = IE_block(1, IE_dim, self.IE_Input_len)
        self.IE_block2 = IE_block(2, IE_dim, self.IE_Input_len)
        self.IE_block3 = IE_block(4, IE_dim, self.IE_Input_len)
        self.IE_block4 = IE_block(8, IE_dim, self.IE_Input_len)
        self.IE_block5 = IE_block(24, IE_dim, self.IE_Input_len)

        self.lay_norm1 = nn.LayerNorm([num_id, self.IE_Input_len, IE_dim])
        self.lay_norm2 = nn.LayerNorm([num_id, self.IE_Input_len, IE_dim])
        self.lay_norm3 = nn.LayerNorm([num_id, self.IE_Input_len, IE_dim])
        self.lay_norm4 = nn.LayerNorm([num_id, self.IE_Input_len, IE_dim])
        self.lay_norm5 = nn.LayerNorm([num_id, self.IE_Input_len, IE_dim])


        # STA-block
        self.ST_block1 = STA_block_att(
            self.IE_Input_len, num_id, IE_dim, out_len, dropout, num_head)
        self.ST_block2 = STA_block_att(
            self.IE_Input_len, num_id, IE_dim, out_len, dropout, num_head)
        self.ST_block3 = STA_block_att(
            self.IE_Input_len, num_id, IE_dim, out_len, dropout, num_head)
        self.ST_block4 = STA_block_att(
            self.IE_Input_len, num_id, IE_dim, out_len, dropout, num_head)
        self.ST_block5 = STA_block_att(
            self.IE_Input_len, num_id, IE_dim, out_len, dropout, num_head)

        # DF_block
        self.DF_block = DF_block(5, out_len)

    def forward(self, history_data, label=None):
        # Input [B,H,N,1] -> Output [B,L,N,1]
        x = history_data[:, :, :, 0]
        x = self.RevIN(x, 'norm').transpose(-2, -1)

        x_day = MGSFformer.Get_Coarse_grain(x, 24)
        x_12h = MGSFformer.Get_Coarse_grain(x, 12)
        x_6h = MGSFformer.Get_Coarse_grain(x, 6)
        x_3h = MGSFformer.Get_Coarse_grain(x, 3)

        # RD-block
        x_day = self.IE_block1(x_day)
        x_12h = self.IE_block2(x_12h)
        x_6h = self.IE_block3(x_6h)
        x_3h = self.IE_block4(x_3h)
        x = self.IE_block5(x)

        x_day = self.lay_norm1(x_day)
        x_12h = self.lay_norm2(x_12h)
        x_6h = self.lay_norm1(x_6h)
        x_3h = self.lay_norm2(x_3h)
        x = self.lay_norm3(x)

        x_12h = x_12h - x_day
        x_6h = x_6h - x_12h
        x_3h = x_3h - x_6h
        x = x - x_3h


        # STA-block
        x_day = self.ST_block1(x_day)
        x_12h = self.ST_block2(x_12h)
        x_6h = self.ST_block3(x_6h)
        x_3h = self.ST_block4(x_3h)
        x = self.ST_block5(x)

        # DF_block
        x_day = x_day.unsqueeze(-1)
        x_12h = x_12h.unsqueeze(-1)
        x_6h = x_6h.unsqueeze(-1)
        x_3h = x_3h.unsqueeze(-1)
        x = x.unsqueeze(-1)
        x = torch.cat([x_day, x_12h, x_6h, x_3h, x], dim=-1)
        x = self.DF_block(x).transpose(-2, -1)
        x = self.RevIN(x, 'denorm').unsqueeze(-1)
        return x

    @staticmethod
    def Get_Coarse_grain(data, n):
        result = 0.0
        for i in range(n):
            line = data[:, :, i::n]
            if i == 0:
                result = line
            else:
                result = result + line
        result = result / n
        return result
