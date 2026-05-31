"""Verbatim MoFo model source.

Vendored from https://github.com/PoorOtterBob/MoFo
(ts_benchmark/baselines/time_series_library/patchs/MoFo.py). Only this header
was added; the model code below is unmodified. The benchmark-facing adapter
lives in ``models.mofo.model``.

Vendored under the upstream project's original license; see THIRD_PARTY_NOTICES.md at the repository root.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MoFo_Backbone(nn.Module):
    def __init__(self, dim, cycle, head):
        super(MoFo_Backbone, self).__init__()
        self.dim = dim
        self.attn = MoFo_Attention(dim, cycle, head)
        self.ffn = SwiGLU_FFN(dim, dim)

        self.attn_norm = RMSNorm(dim, bias=True)
        self.ffn_norm = RMSNorm(dim, bias=True)

    def forward(self, x):
        x = self.attn(self.attn_norm(x)) + x
        x = self.ffn(self.ffn_norm(x)) + x
        return x 

class MoFo_Attention(nn.Module):
    def __init__(self, dim, cycle=24, head=4):
        super(MoFo_Attention, self).__init__()
        # self.dim = dim
        self.head_num = head
        self.head_dim = dim // head
        assert dim % head == 0, "dim must be divisible by head"
        self.transformation = nn.Sequential(
                                nn.Linear(dim, 3*dim),
                                nn.Unflatten(dim=-1, unflattened_size=(self.head_num, 3*self.head_dim))
                            )
        self.outer = nn.Sequential(
                                nn.Flatten(start_dim=-2, end_dim=-1),
                                nn.Linear(dim, dim)
                            )

        self.cycle_in_rotation = nn.Parameter(torch.tensor(2 * torch.pi / cycle), 
                                              requires_grad=False)
        self.cycle_pos = nn.Parameter(torch.arange(0, cycle).unsqueeze(0).unsqueeze(-1).unsqueeze(-1),
                                      requires_grad=False) # (1, 1, cycle)


        self.a_1 = nn.Parameter(torch.zeros(1, cycle))
        self.a_2 = nn.Parameter(torch.zeros(cycle, 1))
        self.b_1 = nn.Parameter(torch.zeros(1, cycle))
        self.b_2 = nn.Parameter(torch.zeros(cycle, 1))
        self.cycle = cycle

        distance = torch.abs(torch.arange(cycle).unsqueeze(1) - torch.arange(cycle).unsqueeze(0))
        self.diff = nn.Parameter(torch.abs(torch.min(distance%cycle, (-distance)%cycle)).float(),
                                 requires_grad=False)
        self.norm = self.head_dim**(-0.5)
    def forward(self, x): 
        query, key, value = torch.chunk(self.transformation(x), 3, dim=-1)
        attention = torch.softmax((query.transpose(1, 2)@key.permute(0, 2, 3, 1)*self.norm + torch.log(self.func())), 
                                  dim=-1) @ value.transpose(1, 2)
        return self.outer(attention.transpose(1, 2))  # -> (B*C, T_C, D)
    
    def RPRope(self, query, key):
        # return query, key
        def chunks(input):
            input_1, input_2 = torch.chunk(input, 2, dim=-1)
            input_inv = torch.cat((-input_2, input_1), dim=-1)
            return input*self.cos_ + input_inv*self.sin_
        return chunks(query), chunks(key)
    
    def func(self):
        a = torch.sigmoid((self.a_1@self.a_2))
        b = torch.sigmoid((self.b_1@self.b_2)) * self.cycle
        return 1/(1+torch.exp(a*(self.diff-b))) + torch.exp(-self.diff)/(1+torch.exp(a*b))

class SwiGLU_FFN(nn.Module):
    def __init__(self, dim_in, dim_out, expand_ratio=4, dropout=0.3, norm=None):
        super(SwiGLU_FFN, self).__init__()
        self.W1 = nn.Linear(dim_in, expand_ratio*dim_in)
        self.W2 = nn.Linear(dim_in, expand_ratio*dim_in)
        self.W3 = nn.Linear(expand_ratio*dim_in, dim_out)
        
        self.dropout =  nn.Dropout(dropout)
        
    def forward(self, x):
        return self.W3(self.dropout(F.silu(self.W1(x)) * self.W2(x)))


class RMSNorm(nn.Module):
    def __init__(self, d, p=-1., eps=1e-8, bias=False):
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0. or self.p > 1.:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1. / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed



class MoFo(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(self, configs, individual=False):
        super(MoFo, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.channels = configs.enc_in
        self.individual = individual
        self.dim = configs.d_model
        self.norm = RevIN(self.channels, eps=1e-5, affine=True)
        self.periodic = configs.periodic
        self.head = configs.head
        self.periodic_index = nn.Parameter(torch.arange(0, self.periodic), requires_grad=False)
        self.periodic_num = math.ceil(self.seq_len/self.periodic)
        try:
            self.layers = configs.d_layers
        except:
            self.layers = 1
        self.if_bias = configs.bias
        self.if_cias = configs.cias
        self.padding_num = self.seq_len % self.periodic

        self.input = nn.Sequential(
                nn.Unflatten(dim=-1, unflattened_size=(self.periodic, self.periodic_num)),
                nn.Linear(self.periodic_num, self.dim),
                )
        # NOTE: the upstream model also built an ``input_multiperiod`` block here.
        # It is never used by the forecasting path, but its parameter
        # initialization advanced the global RNG and thus shifted the init of
        # every layer below. We construct and discard an identical block so the
        # forecast weights are bit-for-bit identical to the upstream model.
        nn.Sequential(
                nn.Unflatten(dim=-1, unflattened_size=(self.periodic, self.periodic_num)),
                nn.Linear(self.periodic_num, self.dim),
                )
        if self.if_bias:
            self.bias = nn.Parameter(torch.empty(1, self.channels, 1, self.dim))
            nn.init.xavier_normal_(self.bias)
        if self.if_cias:
            self.cias = nn.Parameter(torch.empty(self.periodic, self.dim))
            nn.init.xavier_normal_(self.cias)
            # ``ciasW`` was a weekday-aware shift used only by removed task
            # branches. Draw the same RNG so the forecast init is unchanged.
            _ciasW = torch.empty(7, self.dim)
            nn.init.xavier_normal_(_ciasW)
        self.output = nn.Sequential(
                nn.Flatten(start_dim=-2, end_dim=-1),
                nn.Linear(self.dim*self.periodic, self.pred_len),
                )
        # ``regression`` (Linear(periodic, pred_len)) was unused; draw its RNG.
        nn.Linear(self.periodic, self.pred_len)

        self.MoFo_Backbone = nn.Sequential(*[
            MoFo_Backbone(self.dim, self.periodic, self.head) for _ in range(self.layers)
            ])

        
        

    def encoder(self, x, periodic_position, periodic_positionW):
        x = self.norm(x, mode='norm').permute(0, 2, 1) # [Batch, Input length, Channel] -> [Batch*Channel, Input length
        if self.padding_num:
            x = torch.concat([x[..., self.padding_num:self.periodic], x], dim=-1)
            
        x = self.input(x) + self._ias(self.periodic, periodic_position)

        x = self.MoFo_Backbone(x.reshape(-1, self.periodic, self.dim))
        x = self.output(x)
        x = self.norm(x.reshape(-1, self.channels, self.pred_len).permute(0, 2, 1), 
                           mode='denorm')
        return x 

    def _ias(self, p, periodic_position, periodic_positionW=None):
        out = 0
        if self.if_cias:
            c_index = (periodic_position - self.periodic_index.unsqueeze(0))%p # B T_C
            cias = self.cias[c_index.long()].unsqueeze(1)
            out = out + cias
        if self.if_bias:
            out = out + self.bias
        return out


    def forecast(self, x_enc, periodic_position, periodic_positionW):
        # Encoder
        return self.encoder(x_enc, periodic_position, periodic_positionW)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):


        if self.periodic == 24:
            periodic_position = torch.round((x_mark_enc[:, -1, 0:1]+0.5)*(24-1))
        elif self.periodic == 96:
            periodic_position = torch.round((x_mark_enc[:, -1, 2:3]+0.5)*(24-1))*4+\
                                torch.round((x_mark_enc[:, -1, 1:2]+0.5)*(60-1))/15
        elif self.periodic == 144:
            
            periodic_position = torch.round((x_mark_enc[:, -1, 2:3]+0.5)*(24-1))*6+\
                                torch.round((x_mark_enc[:, -1,  1:2]+0.5)*(60-1))/10
        elif self.periodic == 288:
            periodic_position = torch.round((x_mark_enc[:, -1, 2:3]+0.5)*(24-1))*12+\
                                torch.round((x_mark_enc[:, -1, 1:2]+0.5)*(60-1))/5
                                          
        else:
            periodic_position = None
            # raise NotImplementedError
        

        if x_mark_enc.shape[-1] == 4:
                # periodic_positionW = torch.round((x_mark_enc[..., 1:2]+0.5)*(7-1))
                periodic_positionW = x_mark_enc[..., 1]
        elif x_mark_enc.shape[-1] == 6:
            # periodic_positionW = torch.round((x_mark_enc[..., 3:4]+0.5)*(7-1))
            periodic_positionW = x_mark_enc[..., 3]
        else:
            periodic_positionW = None
            # raise NotImplementedError
        

        dec_out = self.forecast(x_enc, periodic_position, periodic_positionW)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]








class Linear(nn.Module):
    def __init__(self, seq_len, pred_len, bias=False):
        super(Linear, self).__init__()
        self.weight = nn.Parameter((1 / seq_len) * torch.ones(1, pred_len, seq_len))
        if bias:
            self.bias = nn.Parameter(torch.zeros(1, pred_len, 1 ))#if self.individual else channels))
        else:
            self.bias = 0
        
    def forward(self, x, relative_cp=1):
        return (relative_cp*self.weight) @ x + self.bias



class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
