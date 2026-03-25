from typing import Optional

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    patch_len: int = 16
    stride: int = 8
    padding_patch: str = "end"
    e_layers: int = 3
    d_model: int = 512
    n_heads: int = 8
    d_k: Optional[int] = None
    d_v: Optional[int] = None
    d_ff: int = 2048
    activation: str = "gelu"
    norm: str = "BatchNorm"
    attn_dropout: float = 0.0
    ffn_dropout: float = 0.0
    res_dropout: float = 0.0
    proj_dropout: float = 0.0
    head_dropout: float = 0.0
    pre_norm: bool = False
    pe: str = "zeros"
    learn_pe: bool = False
    head_type: str = "flatten"
    individual: bool = False
    revin: bool = True
    affine: bool = False
    subtract_last: bool = False
