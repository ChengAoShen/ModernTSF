from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 16
    e_layers: int = 2
    n_heads: int = 8
    d_ff: int = 128
    dropout: float = 0.1
    patch_len: int = 16
    stride: int = 8
    revin: bool = True
    affine: bool = True
    subtract_last: bool = False
    individual: bool = False
    head_dropout: float = 0.0
    cf_dim: int = 48
    cf_depth: int = 2
    cf_heads: int = 6
    cf_mlp: int = 128
    cf_head_dim: int = 32
    cf_drop: float = 0.2
