"""Verbatim CauAir model source.

Vendored from https://github.com/PoorOtterBob/CauAir (src/models/cauair.py).
Only the ``BaseModel`` import path was changed to the shared in-tree base;
the model code below is otherwise unmodified. The benchmark-facing adapter
lives in ``models.cauair.model``.

Vendored under the upstream project's original license; see THIRD_PARTY_NOTICES.md at the repository root.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models._external.base import BaseModel

class CauAir(BaseModel):
    def __init__(self, dim, rank, head=4, **args):
        super(CauAir, self).__init__(**args)
        self.encoder1 = SwiGLU_FFN(self.seq_len, dim)
        self.encoder11 = SwiGLU_FFN(self.seq_len, dim)
        self.encoder2 = SwiGLU_FFN(self.seq_len*(self.input_dim-1), dim)
        self.encoder3 = SwiGLU_FFN(self.seq_len*(self.input_dim-1), dim)

        self.decoder = SwiGLU_FFN(dim, self.output_dim*self.horizon)

        self.position1 = nn.Parameter(torch.zeros((self.node_num, dim)))
        self.position2 = nn.Parameter(torch.zeros((self.node_num, dim)))
        self.position3 = nn.Parameter(torch.zeros((self.node_num, dim)))
        self.position4 = nn.Parameter(torch.zeros((self.node_num, dim)))

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.norm3 = RMSNorm(dim)
        self.norm4 = RMSNorm(dim)

        self.module1 = CachLormer(dim, head, rank, self.node_num)
        self.module2 = CachLormer(dim, head, rank, self.node_num)

        self.alpha = nn.Parameter(torch.tensor(0.))
        self.beta = nn.Parameter(torch.tensor(0.))
    
    def forward(self, x, label=None): 
        z = x[..., 1:].transpose(1, 2).reshape(-1, self.node_num, self.seq_len*(self.input_dim-1))
        z = self.encoder2(z) + self.norm2(self.position2)

        label = label.transpose(1, 2).reshape(-1, self.node_num, self.seq_len*(self.input_dim-1))
        label = self.encoder3(label) +  self.norm3(self.position3)

        x = x[..., 0].transpose(1, -1)
        x1 = self.encoder1(x) +  self.norm1(self.position1)
        x2 = self.encoder11(x) +  self.norm4(self.position4)

        x = F.sigmoid(self.alpha)*self.module1(x1, z) + F.sigmoid(self.beta)*self.module2(x2, label)
        x = self.decoder(x)
        return x.transpose(1, -1).unsqueeze(-1)
    
class CachLormer(nn.Module):
    def __init__(self, dim, head=4, rank=None, node_num=None):
        super(CachLormer, self).__init__()
        self.MHA = Cach_Attention(dim, dim, rank, head)
        self.FFN = SwiGLU_FFN(dim, dim)

        self.norm1 = RMSNorm(dim)

        self.alpha = nn.Parameter(torch.tensor(math.log(9)))
        self.beta = nn.Parameter(torch.tensor(math.log(math.sqrt(2) + 1)))

        self.gamma = nn.Parameter(torch.tensor(math.log(9)))
        self.delta = nn.Parameter(torch.tensor(math.log(math.sqrt(2) + 1)))

    def forward(self, x, z):
        x = self.norm1(F.sigmoid(self.gamma)*x + F.sigmoid(self.delta)*z) # pre-norm
        return F.sigmoid(self.alpha)*self.FFN(x) + F.sigmoid(self.beta)*(self.MHA(x))

class SwiGLU_FFN(nn.Module):
    def __init__(self, dim_in, dim_out, expand_ratio=4, dropout=0.3):
        super(SwiGLU_FFN, self).__init__()
        self.W1 = nn.Linear(dim_in, expand_ratio*dim_in)
        self.W2 = nn.Linear(dim_in, expand_ratio*dim_in)
        self.W3 = nn.Linear(expand_ratio*dim_in, dim_out)
        self.dropout =  nn.Dropout(dropout)

    def forward(self, x):
        return self.W3(self.dropout(F.silu(self.W1(x)) * self.W2(x)))
    

class Cach_Attention(nn.Module):
    def __init__(self, dim, dim_attn, rank, head=4):
        super(Cach_Attention, self).__init__()
        if rank == 0:
            raise 
        else:
            print('cach number is', rank)
        if head == 0:
            raise 
        else:
            print('head number is', head)

        self.head_dim = dim_attn // head
        
        self.query = nn.Linear(dim, dim_attn)
        self.key = nn.Parameter(torch.randn((rank, head, self.head_dim)))
        self.value = nn.Linear(dim, dim)

        self.alpha = nn.Parameter(torch.tensor([math.log(9) for _ in range(head)]).unsqueeze(-1))
        self.beta = nn.Parameter(torch.tensor([math.log(0.01) for _ in range(head)]).unsqueeze(-1))

        self.rank = rank
        self.head = head


    def forward(self, x): 
        b, n, f = x.shape
        q = self.query(x).reshape(b, n, self.head, self.head_dim)
        attn = torch.einsum('bnhd, rhd -> bnhr', q, self.key) / (self.head_dim**0.5)
        x = self.value(x).reshape(b, n, self.head, self.head_dim)
        v = torch.einsum('bnhr, bnhd -> brhd', F.softmax(attn, dim=-1), x)
        v = torch.einsum('bnhr, brhd -> bnhd', F.softmax(attn, dim=-3), v)
        v = F.sigmoid(self.alpha)*x + F.sigmoid(self.beta)*v
        return v.reshape(b, n, f)
    
    
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