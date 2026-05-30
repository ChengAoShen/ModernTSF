from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 128
    n_heads: int = 8
    d_ff: int = 256
    patch_len: int = 16
    stride: int = 8
    dropout: float = 0.1
    factor: int = 3
    activation: str = "gelu"
