from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    n_heads: int = 4
    e_layers: int = 2
    d_ff: int = 128
    seg_len: int = 12
    win_size: int = 2
    factor: int = 10
    dropout: float = 0.1
