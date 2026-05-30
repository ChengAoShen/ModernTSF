import torch

def index_select(input, index, dim=-1):
    if index is list:
        length = len(index)
    else:
        length = index.shape[0]
    
    if length == input.shape[dim]:
        return input
    else:
        return torch.index_select(input, index=index, dim=dim)
    