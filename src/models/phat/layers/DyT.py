import torch
import torch.nn as nn
import torch.nn.functional as F


class DyT(nn.Module):
    def __init__(self, dim, init_alpha=0.5):
        super(DyT, self).__init__()
        self.alpha = nn.Parameter(torch.tensor([init_alpha]))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.gamma * F.tanh(self.alpha * x) + self.beta