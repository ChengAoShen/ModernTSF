import torch
import torch.nn as nn

class Transformer_Block(nn.Module):
    def __init__(self, Attn, FFN, Attn_Norm, FFN_Norm, Pre_Norm=True):
        super(Transformer_Block, self).__init__()
        self.Pre_Norm = Pre_Norm
        self.attn = Attn
        self.ffn = FFN
        self.attn_norm = Attn_Norm
        self.ffn_norm = FFN_Norm

    def forward(self, x):
        if self.Pre_Norm:
            x = self.attn(self.attn_norm(x)) + x
            x = self.ffn(self.ffn_norm(x)) + x
        else:
            x = self.attn_norm(self.attn(x) + x)
            x = self.ffn_norm(self.ffn(x) + x)
        return x 