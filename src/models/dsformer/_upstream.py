"""Upstream DSFormer model from CauAir.

Bundles dsformer.py + TVA_block.py + decoder_block.py into one file.
BaseModel replaced by nn.Module with explicit params.
"""

import torch
from torch import nn
import torch.nn.functional as F


# ============================================================
# TVA_block.py classes
# ============================================================

class Time_att(nn.Module):
    def __init__(self, dim_input, dropout, num_head):
        super(Time_att, self).__init__()
        self.query = nn.Conv2d(in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.key = nn.Conv2d(in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.value = nn.Conv2d(in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.laynorm = nn.LayerNorm([dim_input])
        self.softmax = nn.Softmax(dim=-1)
        self.num_head = num_head
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(num_head, 1)

    def forward(self, x):
        x = x.transpose(-3, -1)
        result = 0.0
        for i in range(self.num_head):
            q = self.dropout(self.query(x)).transpose(-3, -1)
            k = self.dropout(self.key(x)).transpose(-3, -1)
            k = k.transpose(-2, -1)
            v = self.dropout(self.value(x)).transpose(-3, -1)
            kd = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32) / self.num_head)
            line = self.dropout(self.softmax(q @ k / kd)) @ v
            if i < 1:
                result = line.unsqueeze(-1)
            else:
                result = torch.cat([result, line.unsqueeze(-1)], dim=-1)
        result = self.output(result)
        result = result.squeeze(-1)
        x = x.transpose(-3, -1) + result
        x = self.laynorm(x)
        return x

class space_att_enc(nn.Module):
    """space_attention2 from TVA_block.py (encoder version)."""
    def __init__(self, Input_len, dim_input, dropout, num_head):
        super(space_att_enc, self).__init__()
        self.query = nn.Linear(dim_input, dim_input)
        self.key = nn.Linear(dim_input, dim_input)
        self.value = nn.Linear(dim_input, dim_input)
        self.softmax = nn.Softmax(dim=-1)
        self.num_head = num_head
        self.linear1 = nn.Linear(num_head, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 3)
        result = 0.0
        q = self.dropout(self.query(x))
        k = self.dropout(self.key(x))
        k = k.transpose(-2, -1)
        v = self.dropout(self.value(x))
        kd = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32) / self.num_head)
        for i in range(self.num_head):
            line = self.dropout(self.softmax(q @ k / kd)) @ v
            if i < 1:
                result = line.unsqueeze(-1)
            else:
                result = torch.cat([result, line.unsqueeze(-1)], dim=-1)
        result = self.linear1(result)
        result = result.squeeze(-1)
        result = result.transpose(1, 3)
        return result


class cross_att(nn.Module):
    def __init__(self, dim_input, dropout, num_head):
        super(cross_att, self).__init__()
        self.query = nn.Conv2d(in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.key = nn.Conv2d(in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.value = nn.Conv2d(in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.laynorm = nn.LayerNorm([dim_input])
        self.softmax = nn.Softmax(dim=-1)
        self.num_head = num_head
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(num_head, 1)

    def forward(self, x, x2):
        x = x.transpose(-3, -1)
        x2 = x2.transpose(-3, -1)
        result = 0.0
        for i in range(self.num_head):
            q = self.dropout(self.query(x2)).transpose(-3, -1)
            k = self.dropout(self.key(x)).transpose(-3, -1)
            k = k.transpose(-2, -1)
            v = self.dropout(self.value(x)).transpose(-3, -1)
            kd = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32) / self.num_head)
            line = self.dropout(self.softmax(q @ k / kd)) @ v
            if i < 1:
                result = line.unsqueeze(-1)
            else:
                result = torch.cat([result, line.unsqueeze(-1)], dim=-1)
        result = self.output(result)
        result = result.squeeze(-1)
        x = x.transpose(-3, -1) + result
        x = self.laynorm(x)
        return x


class TVA_block_att(nn.Module):
    def __init__(self, Input_len, num_id, num_layer, dropout, num_head, num_samp):
        super(TVA_block_att, self).__init__()
        self.num_lay = num_layer
        self.Time_att = Time_att(Input_len, dropout, num_head)
        self.space_att = space_att_enc(Input_len, num_id, dropout, num_head)
        self.cross_att = cross_att(Input_len, dropout, num_head)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Conv2d(in_channels=Input_len, out_channels=Input_len, kernel_size=(num_samp, 1))

    def forward(self, x):
        for i in range(self.num_lay):
            x = self.cross_att(self.Time_att(x), self.space_att(x))
        x = self.linear(x.transpose(-3, -1))
        x = x.squeeze(-2)
        return x.transpose(-2, -1)


# ============================================================
# decoder_block.py classes
# ============================================================

class Time_de_att(nn.Module):
    def __init__(self, dim_input, dropout, num_head):
        super(Time_de_att, self).__init__()
        self.query = nn.Conv1d(in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.key = nn.Conv1d(in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.value = nn.Conv1d(in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.laynorm = nn.LayerNorm([dim_input])
        self.softmax = nn.Softmax(dim=-1)
        self.num_head = num_head
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Conv2d(
            in_channels=dim_input, out_channels=dim_input, kernel_size=(1, num_head))

    def forward(self, x):
        x = x.transpose(-2, -1)
        result = 0.0
        for i in range(self.num_head):
            q = self.dropout(self.query(x)).transpose(-2, -1)
            k = self.dropout(self.key(x))
            v = self.dropout(self.value(x)).transpose(-2, -1)
            kd = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32) / self.num_head)
            line = self.dropout(self.softmax(q @ k / kd)) @ v
            if i < 1:
                result = line.unsqueeze(-1)
            else:
                result = torch.cat([result, line.unsqueeze(-1)], dim=-1)
        result = self.output(result.transpose(1, 2))
        result = result.squeeze(-1)
        x = x + result
        x = x.transpose(-2, -1)
        x = self.laynorm(x)
        return x


class space_att_dec(nn.Module):
    """space_attention2 from decoder_block.py."""
    def __init__(self, Input_len, dim_input, dropout, num_head):
        super(space_att_dec, self).__init__()
        self.query = nn.Linear(dim_input, dim_input)
        self.key = nn.Linear(dim_input, dim_input)
        self.value = nn.Linear(dim_input, dim_input)
        self.softmax = nn.Softmax(dim=-1)
        self.num_head = num_head
        self.linear1 = nn.Linear(num_head, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)
        result = 0.0
        q = self.dropout(self.query(x))
        k = self.dropout(self.key(x))
        k = k.transpose(-2, -1)
        v = self.dropout(self.value(x))
        kd = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32) / self.num_head)
        for i in range(self.num_head):
            line = self.dropout(self.softmax(q @ k / kd)) @ v
            if i < 1:
                result = line.unsqueeze(-1)
            else:
                result = torch.cat([result, line.unsqueeze(-1)], dim=-1)
        result = self.linear1(result)
        result = result.squeeze(-1)
        result = result.transpose(1, 2)
        return result


class cross_de_att(nn.Module):
    def __init__(self, dim_input, dropout, num_head):
        super(cross_de_att, self).__init__()
        self.query = nn.Conv1d(in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.key = nn.Conv1d(in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.value = nn.Conv1d(in_channels=dim_input, out_channels=dim_input, kernel_size=1)
        self.laynorm = nn.LayerNorm([dim_input])
        self.softmax = nn.Softmax(dim=-1)
        self.num_head = num_head
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Conv2d(
            in_channels=dim_input, out_channels=dim_input, kernel_size=(1, num_head))

    def forward(self, x, x2):
        x = x.transpose(-2, -1)
        x2 = x2.transpose(-2, -1)
        result = 0.0
        for i in range(self.num_head):
            q = self.dropout(self.query(x2)).transpose(-2, -1)
            k = self.dropout(self.key(x))
            v = self.dropout(self.value(x)).transpose(-2, -1)
            kd = torch.sqrt(torch.tensor(k.shape[-1]).to(torch.float32) / self.num_head)
            line = self.dropout(self.softmax(q @ k / kd)) @ v
            if i < 1:
                result = line.unsqueeze(-1)
            else:
                result = torch.cat([result, line.unsqueeze(-1)], dim=-1)
        result = self.output(result.transpose(1, 2))
        result = result.squeeze(-1)
        x = x + result
        x = x.transpose(-2, -1)
        x = self.laynorm(x)
        return x


class TVADE_block(nn.Module):
    def __init__(self, Input_len, num_id, dropout, num_head=1):
        super(TVADE_block, self).__init__()
        self.Time_att = Time_de_att(Input_len, dropout, num_head)
        self.space_att = space_att_dec(Input_len, num_id, dropout, num_head)
        self.cross_att = cross_de_att(Input_len, dropout, num_head)

    def forward(self, x):
        x = self.cross_att(self.Time_att(x), self.space_att(x))
        return x


# ============================================================
# dsformer.py main model + helpers
# ============================================================

class embed(nn.Module):
    def __init__(self, Input_len, num_id, num_samp, IF_node):
        super(embed, self).__init__()
        self.IF_node = IF_node
        self.num_samp = num_samp
        self.embed_layer = nn.Linear(2 * Input_len, Input_len)
        self.node_emb = nn.Parameter(torch.empty(num_id, Input_len))
        nn.init.xavier_uniform_(self.node_emb)

    def forward(self, x):
        x = x.unsqueeze(-1)
        batch_size, _, _, _ = x.shape
        node_emb1 = self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(-1)

        x_1 = embed.down_sampling(x, self.num_samp)
        if self.IF_node:
            x_1 = torch.cat([x_1, embed.down_sampling(node_emb1, self.num_samp)], dim=-1)

        x_2 = embed.Interval_sample(x, self.num_samp)
        if self.IF_node:
            x_2 = torch.cat([x_2, embed.Interval_sample(node_emb1, self.num_samp)], dim=-1)

        return x_1, x_2

    @staticmethod
    def down_sampling(data, n):
        result = 0.0
        for i in range(n):
            line = data[:, :, i::n, :]
            if i == 0:
                result = line
            else:
                result = torch.cat([result, line], dim=3)
        result = result.transpose(2, 3)
        return result

    @staticmethod
    def Interval_sample(data, n):
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


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
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
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

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


class DSFormer(nn.Module):
    def __init__(self, node_num: int, seq_len: int, horizon: int,
                 num_layer: int = 1, dropout: float = 0.2,
                 muti_head: int = 4, num_samp: int = 3,
                 IF_node: bool = True):
        super(DSFormer, self).__init__()
        self.node_num = node_num
        self.seq_len = seq_len
        self.horizon = horizon

        Input_len = self.seq_len
        out_len = self.horizon
        num_id = self.node_num

        if IF_node:
            self.inputlen = 2 * Input_len // num_samp
        else:
            self.inputlen = Input_len // num_samp

        # embed and encoder
        self.RevIN = RevIN(num_id)
        self.embed_layer = embed(Input_len, num_id, num_samp, IF_node)
        self.encoder = TVA_block_att(
            self.inputlen, num_id, num_layer, dropout, muti_head, num_samp)
        self.laynorm = nn.LayerNorm([self.inputlen])

        # decoder
        self.decoder = TVADE_block(self.inputlen, num_id, dropout, muti_head)
        self.output = nn.Conv1d(
            in_channels=self.inputlen, out_channels=out_len, kernel_size=1)

    def forward(self, x, label=None):
        # Input [B, H, N]: B batch, N variables, H history length
        # Output [B, L, N]: L future length
        x = self.RevIN(x, 'norm').transpose(-2, -1)
        x_1, x_2 = self.embed_layer(x)

        # encoder
        x_1 = self.encoder(x_1)
        x_2 = self.encoder(x_2)
        x = x_1 + x_2
        x = self.laynorm(x)

        # decoder
        x = self.decoder(x)
        x = self.output(x.transpose(-2, -1))
        x = self.RevIN(x, 'denorm')
        return x

