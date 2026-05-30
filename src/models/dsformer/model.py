"""DSFormer model implementation.

Vendored/adapted from https://github.com/GestaltCogTeam/DSformer
(main_model.py, block/embed_block.py, block/TVA_block.py,
block/decoder_block.py). The upstream repository declares NO LICENSE
(no LICENSE file, none stated in the README) — i.e. all rights reserved
by the authors. It is included here for research/benchmarking parity; see
THIRD_PARTY_NOTICES for the recorded provenance.

DSformer: A Double Sampling Transformer for Multivariate Time Series
Long-term Prediction (CIKM 2023).

Adapted for ModernTSF: the upstream standalone/BasicTS-style constructor
``DSFormer(Input_len, out_len, num_id, num_layer, dropout, muti_head,
num_samp, IF_node)`` and ``forward(x)`` (input ``[B, H, N]``) are wrapped to
the ModernTSF plain-kwargs constructor and the
``forward(x_enc, x_mark_enc, x_dec, x_mark_dec)`` contract returning
``(B, pred_len, c_out)``. The double-sampling embedding and the
TVA encoder / decoder attention blocks are DSformer-specific and kept local
to this file. The shared ``RevIN`` layer under ``models.module.revin`` is
reused (it is byte-identical to the upstream block/revin.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.module.revin import RevIN


# --------------------------------------------------------------------------- #
# Double-sampling embedding
# --------------------------------------------------------------------------- #
class Embed(nn.Module):
    def __init__(self, input_len, num_id, num_samp, if_node):
        super().__init__()
        self.if_node = if_node
        self.num_samp = num_samp
        self.node_emb = nn.Parameter(torch.empty(num_id, input_len))
        nn.init.xavier_uniform_(self.node_emb)

    def forward(self, x):
        x = x.unsqueeze(-1)
        batch_size = x.shape[0]
        node_emb1 = (
            self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(-1)
        )

        x_1 = self.down_sampling(x, self.num_samp)
        if self.if_node:
            x_1 = torch.cat(
                [x_1, self.down_sampling(node_emb1, self.num_samp)], dim=-1
            )

        x_2 = self.interval_sample(x, self.num_samp)
        if self.if_node:
            x_2 = torch.cat(
                [x_2, self.interval_sample(node_emb1, self.num_samp)], dim=-1
            )

        return x_1, x_2

    @staticmethod
    def down_sampling(data, n):
        result = 0.0
        for i in range(n):
            line = data[:, :, i::n, :]
            result = line if i == 0 else torch.cat([result, line], dim=3)
        return result.transpose(2, 3)

    @staticmethod
    def interval_sample(data, n):
        result = 0.0
        data_len = data.shape[2] // n
        for i in range(n):
            line = data[:, :, data_len * i : data_len * (i + 1), :]
            result = line if i == 0 else torch.cat([result, line], dim=3)
        return result.transpose(2, 3)


# --------------------------------------------------------------------------- #
# Encoder attention blocks (TVA block)
# --------------------------------------------------------------------------- #
class TimeAtt(nn.Module):
    def __init__(self, dim_input, dropout, num_head):
        super().__init__()
        self.query = nn.Conv2d(dim_input, dim_input, kernel_size=1)
        self.key = nn.Conv2d(dim_input, dim_input, kernel_size=1)
        self.value = nn.Conv2d(dim_input, dim_input, kernel_size=1)
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
            k = self.dropout(self.key(x)).transpose(-3, -1).transpose(-2, -1)
            v = self.dropout(self.value(x)).transpose(-3, -1)
            kd = torch.sqrt(
                torch.tensor(k.shape[-1]).to(torch.float32) / self.num_head
            )
            line = self.dropout(self.softmax(q @ k / kd)) @ v
            result = (
                line.unsqueeze(-1)
                if i < 1
                else torch.cat([result, line.unsqueeze(-1)], dim=-1)
            )
        result = self.output(result).squeeze(-1)
        x = x.transpose(-3, -1) + result
        return self.laynorm(x)


class SpaceAtt2(nn.Module):
    def __init__(self, input_len, dim_input, dropout, num_head):
        super().__init__()
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
        k = self.dropout(self.key(x)).transpose(-2, -1)
        v = self.dropout(self.value(x))
        kd = torch.sqrt(
            torch.tensor(k.shape[-1]).to(torch.float32) / self.num_head
        )
        for i in range(self.num_head):
            line = self.dropout(self.softmax(q @ k / kd)) @ v
            result = (
                line.unsqueeze(-1)
                if i < 1
                else torch.cat([result, line.unsqueeze(-1)], dim=-1)
            )
        result = self.linear1(result).squeeze(-1)
        return result.transpose(1, 3)


class CrossAtt(nn.Module):
    def __init__(self, dim_input, dropout, num_head):
        super().__init__()
        self.query = nn.Conv2d(dim_input, dim_input, kernel_size=1)
        self.key = nn.Conv2d(dim_input, dim_input, kernel_size=1)
        self.value = nn.Conv2d(dim_input, dim_input, kernel_size=1)
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
            k = self.dropout(self.key(x)).transpose(-3, -1).transpose(-2, -1)
            v = self.dropout(self.value(x)).transpose(-3, -1)
            kd = torch.sqrt(
                torch.tensor(k.shape[-1]).to(torch.float32) / self.num_head
            )
            line = self.dropout(self.softmax(q @ k / kd)) @ v
            result = (
                line.unsqueeze(-1)
                if i < 1
                else torch.cat([result, line.unsqueeze(-1)], dim=-1)
            )
        result = self.output(result).squeeze(-1)
        x = x.transpose(-3, -1) + result
        return self.laynorm(x)


class TVABlockAtt(nn.Module):
    def __init__(self, input_len, num_id, num_layer, dropout, num_head, num_samp):
        super().__init__()
        self.num_lay = num_layer
        self.time_att = TimeAtt(input_len, dropout, num_head)
        self.space_att = SpaceAtt2(input_len, num_id, dropout, num_head)
        self.cross_att = CrossAtt(input_len, dropout, num_head)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Conv2d(
            input_len, input_len, kernel_size=(num_samp, 1)
        )

    def forward(self, x):
        for _ in range(self.num_lay):
            x = self.cross_att(self.time_att(x), self.space_att(x))
        x = self.linear(x.transpose(-3, -1))
        x = x.squeeze(-2)
        return x.transpose(-2, -1)


# --------------------------------------------------------------------------- #
# Decoder attention blocks (TVADE block)
# --------------------------------------------------------------------------- #
class TimeDeAtt(nn.Module):
    def __init__(self, dim_input, dropout, num_head):
        super().__init__()
        self.query = nn.Conv1d(dim_input, dim_input, kernel_size=1)
        self.key = nn.Conv1d(dim_input, dim_input, kernel_size=1)
        self.value = nn.Conv1d(dim_input, dim_input, kernel_size=1)
        self.laynorm = nn.LayerNorm([dim_input])
        self.softmax = nn.Softmax(dim=-1)
        self.num_head = num_head
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Conv2d(dim_input, dim_input, kernel_size=(1, num_head))

    def forward(self, x):
        x = x.transpose(-2, -1)
        result = 0.0
        for i in range(self.num_head):
            q = self.dropout(self.query(x)).transpose(-2, -1)
            k = self.dropout(self.key(x))
            v = self.dropout(self.value(x)).transpose(-2, -1)
            kd = torch.sqrt(
                torch.tensor(k.shape[-1]).to(torch.float32) / self.num_head
            )
            line = self.dropout(self.softmax(q @ k / kd)) @ v
            result = (
                line.unsqueeze(-1)
                if i < 1
                else torch.cat([result, line.unsqueeze(-1)], dim=-1)
            )
        result = self.output(result.transpose(1, 2)).squeeze(-1)
        x = x + result
        x = x.transpose(-2, -1)
        return self.laynorm(x)


class SpaceDeAtt2(nn.Module):
    def __init__(self, input_len, dim_input, dropout, num_head):
        super().__init__()
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
        k = self.dropout(self.key(x)).transpose(-2, -1)
        v = self.dropout(self.value(x))
        kd = torch.sqrt(
            torch.tensor(k.shape[-1]).to(torch.float32) / self.num_head
        )
        for i in range(self.num_head):
            line = self.dropout(self.softmax(q @ k / kd)) @ v
            result = (
                line.unsqueeze(-1)
                if i < 1
                else torch.cat([result, line.unsqueeze(-1)], dim=-1)
            )
        result = self.linear1(result).squeeze(-1)
        return result.transpose(1, 2)


class CrossDeAtt(nn.Module):
    def __init__(self, dim_input, dropout, num_head):
        super().__init__()
        self.query = nn.Conv1d(dim_input, dim_input, kernel_size=1)
        self.key = nn.Conv1d(dim_input, dim_input, kernel_size=1)
        self.value = nn.Conv1d(dim_input, dim_input, kernel_size=1)
        self.laynorm = nn.LayerNorm([dim_input])
        self.softmax = nn.Softmax(dim=-1)
        self.num_head = num_head
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Conv2d(dim_input, dim_input, kernel_size=(1, num_head))

    def forward(self, x, x2):
        x = x.transpose(-2, -1)
        x2 = x2.transpose(-2, -1)
        result = 0.0
        for i in range(self.num_head):
            q = self.dropout(self.query(x2)).transpose(-2, -1)
            k = self.dropout(self.key(x))
            v = self.dropout(self.value(x)).transpose(-2, -1)
            kd = torch.sqrt(
                torch.tensor(k.shape[-1]).to(torch.float32) / self.num_head
            )
            line = self.dropout(self.softmax(q @ k / kd)) @ v
            result = (
                line.unsqueeze(-1)
                if i < 1
                else torch.cat([result, line.unsqueeze(-1)], dim=-1)
            )
        result = self.output(result.transpose(1, 2)).squeeze(-1)
        x = x + result
        x = x.transpose(-2, -1)
        return self.laynorm(x)


class TVADEBlock(nn.Module):
    def __init__(self, input_len, num_id, dropout, num_head=1):
        super().__init__()
        self.time_att = TimeDeAtt(input_len, dropout, num_head)
        self.space_att = SpaceDeAtt2(input_len, num_id, dropout, num_head)
        self.cross_att = CrossDeAtt(input_len, dropout, num_head)

    def forward(self, x):
        return self.cross_att(self.time_att(x), self.space_att(x))


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        enc_in,
        label_len=0,
        features="M",
        num_layer=1,
        muti_head=2,
        num_samp=2,
        dropout=0.15,
        if_node=True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features
        self.num_id = enc_in

        if if_node:
            self.inputlen = 2 * seq_len // num_samp
        else:
            self.inputlen = seq_len // num_samp

        self.revin = RevIN(enc_in)
        self.embed_layer = Embed(seq_len, enc_in, num_samp, if_node)
        self.encoder = TVABlockAtt(
            self.inputlen, enc_in, num_layer, dropout, muti_head, num_samp
        )
        self.laynorm = nn.LayerNorm([self.inputlen])
        self.decoder = TVADEBlock(self.inputlen, enc_in, dropout, muti_head)
        self.output = nn.Conv1d(self.inputlen, pred_len, kernel_size=1)

    def forecast(self, x_enc):
        # x_enc: [B, seq_len, N]
        x = self.revin(x_enc, "norm").transpose(-2, -1)  # [B, N, seq_len]
        x_1, x_2 = self.embed_layer(x)

        x_1 = self.encoder(x_1)
        x_2 = self.encoder(x_2)
        x = x_1 + x_2
        x = self.laynorm(x)

        x = self.decoder(x)
        x = self.output(x.transpose(-2, -1))  # [B, pred_len, N]
        x = self.revin(x, "denorm")
        return x

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        dec_out = self.forecast(x_enc)
        return dec_out[:, -self.pred_len :, :]  # [B, pred_len, c_out]
