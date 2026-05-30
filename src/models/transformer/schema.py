from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    dec_in: int | None = None
    c_out: int | None = None
    d_model: int = 128
    n_heads: int = 8
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 256
    dropout: float = 0.1
    factor: int = 3
    activation: str = "gelu"
    embed: str = "timeF"
    freq: str = "h"
