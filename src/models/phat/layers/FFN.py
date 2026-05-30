import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.phat.layers.Function import index_select


class Linear(nn.Linear):
    def __init__(self, in_features, out_features, n_vars=None, bias=True, device=None, dtype=None):
        if n_vars is None:
            super().__init__(in_features, out_features, bias, device, dtype)
            self.n_vars = None
        else:
            self.n_vars = n_vars
            factory_kwargs = {'device': device, 'dtype': dtype}
            super(nn.Linear, self).__init__()
            self.weight = nn.Parameter(torch.empty((n_vars, in_features, out_features), **factory_kwargs))
            if bias:
                self.bias = nn.Parameter(torch.empty((n_vars, out_features), **factory_kwargs))
            else:
                self.register_parameter('bias', None)
            self.reset_parameters()

    def forward(self, x):
        if self.n_vars is None:
            return super().forward(x)
        y = torch.einsum('...vi, vio -> ...vo', x, self.weight)
        if self.bias is not None:
            y = y + self.bias
        return y

class Channel_Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        # print('!!!!!!!!!!!!!!!!!!!', self.weight.shape, '!!!!!!!!!!!!!')

    def forward(self, x, channel_index=None, encoder=True):
        if channel_index is None:
            return super().forward(x)
        if encoder:
            y = F.linear(index_select(x, channel_index, dim=-1),  
                         index_select(self.weight, channel_index, dim=-1), 
                         self.bias)
        else:
            y = F.linear(x, index_select(self.weight, channel_index, dim=0))
        return y

    def first_forward(self, x):
        y = x[..., None, :] * self.weight # (...i) -> (...io)
        return y
    
    def second_forward(self, x, index, dim=-1):
        return self.add_bias(torch.sum(index_select(x, index, dim=dim), dim=dim))
    
    def add_bias(self, x, channel_index=None):
        if self.bias is None:
            return x
        else:
            x = x + self.bias if channel_index is None else index_select(self.bias, channel_index, dim=0)
        return x



class SwiGLU_FFN(nn.Module):
    def __init__(self, dim_in, dim_out, dropout=0.3, expand_ratio=2.67):
        super(SwiGLU_FFN, self).__init__()

        self.W1 = nn.Linear(dim_in, round(expand_ratio*dim_in))
        self.W2 = nn.Linear(dim_in, round(expand_ratio*dim_in))
        self.W3 = nn.Linear(round(expand_ratio*dim_in), dim_out)
        self.dropout =  nn.Dropout(dropout)

    def forward(self, x):
        return self.W3(self.dropout(F.silu(self.W1(x)) * self.W2(x)))
    

