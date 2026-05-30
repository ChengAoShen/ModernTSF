from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 32
    e_layers: int = 1
    patch_len: int = 16
    stride: int = 8
    heads: int = 2
    dropout: float = 0.1
    moving_avg: int = 25
    expert_num: int = 2
    kan_div: int = 4
    k: int = 1
    aggregated_norm: int = 1
