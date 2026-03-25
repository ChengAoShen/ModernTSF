"""Positional encoding utilities for PatchTST."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def PositionalEncoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return pe


def Coord1dPosEncoding(q_len, exponential=False, normalize=True):
    cpe = (
        2 * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** (0.5 if exponential else 1))
        - 1
    )
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def Coord2dPosEncoding(
    q_len, d_model, exponential=False, normalize=True, eps=1e-3, verbose=False
):
    x = 0.5 if exponential else 1
    for _ in range(100):
        cpe = (
            2
            * (torch.linspace(0, 1, q_len).reshape(-1, 1) ** x)
            * (torch.linspace(0, 1, d_model).reshape(1, -1) ** x)
            - 1
        )
        if abs(cpe.mean()) <= eps:
            break
        if cpe.mean() > eps:
            x += 0.001
        else:
            x -= 0.001
    if normalize:
        cpe = cpe - cpe.mean()
        cpe = cpe / (cpe.std() * 10)
    return cpe


def positional_encoding(pe, learn_pe, q_len, d_model):
    if pe is None:
        w_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(w_pos, -0.02, 0.02)
        learn_pe = False
    elif pe == "zero":
        w_pos = torch.empty((q_len, 1))
        nn.init.uniform_(w_pos, -0.02, 0.02)
    elif pe == "zeros":
        w_pos = torch.empty((q_len, d_model))
        nn.init.uniform_(w_pos, -0.02, 0.02)
    elif pe in {"normal", "gauss"}:
        w_pos = torch.zeros((q_len, 1))
        nn.init.normal_(w_pos, mean=0.0, std=0.1)
    elif pe == "uniform":
        w_pos = torch.zeros((q_len, 1))
        nn.init.uniform_(w_pos, a=0.0, b=0.1)
    elif pe == "lin1d":
        w_pos = Coord1dPosEncoding(q_len, exponential=False, normalize=True)
    elif pe == "exp1d":
        w_pos = Coord1dPosEncoding(q_len, exponential=True, normalize=True)
    elif pe == "lin2d":
        w_pos = Coord2dPosEncoding(q_len, d_model, exponential=False, normalize=True)
    elif pe == "exp2d":
        w_pos = Coord2dPosEncoding(q_len, d_model, exponential=True, normalize=True)
    elif pe == "sincos":
        w_pos = PositionalEncoding(q_len, d_model, normalize=True)
    else:
        raise ValueError(
            f"{pe} is not a valid pe (positional encoder. Available types: 'gauss'=='normal', "
            "'zeros', 'zero', 'uniform', 'lin1d', 'exp1d', 'lin2d', 'exp2d', 'sincos', None.)"
        )

    return nn.Parameter(w_pos, requires_grad=bool(learn_pe))
