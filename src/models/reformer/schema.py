from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    c_out: int | None = None
    d_model: int = 128
    n_heads: int = 8
    e_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    activation: str = "gelu"
    embed: str = "timeF"
    freq: str = "h"
    bucket_size: int = 4
    n_hashes: int = 4
