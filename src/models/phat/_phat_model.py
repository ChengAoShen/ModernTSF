"""Verbatim PHAT model source.

Vendored from https://github.com/PoorOtterBob/PHAT (phat/models/phat_model.py).
Only the import paths were rewritten to the in-tree package; the model code
below is otherwise unmodified. The benchmark-facing adapter lives in
``models.phat.model``.

Note: the upstream repository ships ``phat_model.py`` but omits its core
``PHAT_Attention`` module (the Positive-Negative X-shape Attention). It is
reconstructed from the paper (ICLR 2026, arXiv:2602.00654) in
``models.phat.layers.PHAT_Attention``; see that file for the equation mapping.

Vendored under the upstream project's original license; see THIRD_PARTY_NOTICES.md at the repository root.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.phat.layers.DyT import DyT
from models.phat.layers.RevIN import RevIN
from models.phat.layers.Transformer_Block import Transformer_Block
from models.phat.layers.PHAT_Attention import PHAT_Attention
from models.phat.layers.FFN import SwiGLU_FFN, Linear, Channel_Linear
from models.phat.layers.Function import index_select


class PHATModel(nn.Module):
    def __init__(self, config):
        super(PHATModel, self).__init__()
        self.CI = config.CI
        print('There is' + str(' ' if self.CI else ' not ') + 'channel individual!' )
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        # self.series_len = self.seq_len + self.pred_len
        self.series_len = self.pred_len

        if config.period_list is not None:
            self.register_buffer('period_list', torch.tensor(config.period_list))
        else:
            self.period_list = None
        self.period_topk = config.period_topk # if self.period_list is None else len(self.period_list)
        print('Period Top K is', self.period_topk, "!")


        self.d_layers = config.d_layers
        self.d_model = config.d_model
        self.head = config.n_heads
        self.attn_dropout = config.attn_dropout

        self.ffn_dropout = config.ffn_dropout
        self.ffn_expand_ratio = config.ffn_expand_ratio

        self.n_vars = config.enc_in
        self.output_base_pred = config.output_base_pred

        self.Normalization = RevIN(config.enc_in)
        self.base_pred = Linear(self.seq_len, self.pred_len, self.n_vars if self.CI else None)
        self.encoder = Channel_Linear(self.n_vars, self.d_model)
        self.decoder = Channel_Linear(self.d_model, self.n_vars)

        self.backbone = nn.Sequential(
            *[
                Transformer_Block(
                    PHAT_Attention(self.d_model, self.head, self.attn_dropout, l, 
                                    ),
                    SwiGLU_FFN(self.d_model, self.d_model, self.ffn_dropout, self.ffn_expand_ratio),
                    DyT(self.d_model),
                    DyT(self.d_model),
                ) for l in range(self.d_layers)
            ],
            nn.Flatten(start_dim=1, end_dim=2),
        )

    def forward(self, x, pre_train=False):
        # x: [batch_size, seq_len, n_vars]
        x = self.Normalization(x, 'norm')
        x = self._forecast(x, pre_train)
        x = self.Normalization(x, 'denorm')
        
        return x

    
    def _forecast(self, x, pre_train):
        x_ = self.base_pred(x.transpose(-1, -2)).transpose(-1, -2)
        if pre_train:
            return x_
        x = self._period_transformer_pipeline(x_, x) + x_
        return x*0.5

    def _period_transformer_pipeline(self, x, period_cal):
        period_list, periods_index_list, weight_list = self._fft_for_period(period_cal)
            
        y = torch.zeros_like(x, device=x.device) #  (B, T, N)
        x = self.encoder.first_forward(x) # (B, T, D, N)
        
        for period, period_index, weight in zip(period_list, periods_index_list, weight_list):
            for p, index in zip(period, period_index):
                if index.shape[0] == 0:
                    break

                w = index_select(weight, index, dim=-1)

                if torch.mean(w) < 2e-2:
                    break
                x_ = self.encoder.second_forward(x, index)

                x_, series_len = self._padding(x_, p)
                x_ = x_.reshape(x.shape[0], p, series_len // p, self.d_model).contiguous()
                x_ = self.backbone(x_)[:, :self.series_len]
                x_ = self.decoder.forward(x_, index, encoder=False)
                y[..., index] += x_ * torch.mean(w[..., None], dim=-2, keepdim=True)
        
        y = self.decoder.add_bias(y)
        return y

    def _fft_for_period(self, x):
        # [B, T, C]
        xf = torch.fft.rfft(x, dim=1)
        # find period by amplitudes
        frequency_list = abs(xf).mean(0) # [T', C]
        frequency_list[0] = 0
        _, top_list = torch.topk(frequency_list, self.period_topk, dim=-2)
        periods = torch.round(x.shape[1] / top_list).long()
        periods[periods >= self.seq_len] = self.series_len
        periods_list = []
        for period in periods:
            periods_list.append(torch.unique(period) if self.period_list is None else self.period_list)
        periods_index_list = [[torch.where(periods == p)[-1] for p in top_p] for top_p in periods_list]
        weight_list = torch.gather(abs(xf), dim=1, index=top_list[None].expand(x.shape[0], -1, -1)).transpose(0, 1)
        weight_list = torch.softmax(weight_list, dim=0)

        return periods_list, periods_index_list, weight_list

    def _padding(self, x, period):
        padding_num = (period - self.series_len % period) % period
        if padding_num:
            # B L C -> B L+T C
            out = F.pad(x, (0, 0, 0, padding_num), 'constant', 0)
        else:
            out = x
        
        return out, self.series_len + padding_num



