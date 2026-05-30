from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    patch_len: int = 16
    stride: int = 8
    d_model: int = 128
    n_heads: int = 8
    e_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    dp_rank: int = 8
    merge_size: int = 2
    momentum: float = 0.1
    alpha: float = 0.5
    use_statistic: bool = False
