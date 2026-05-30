from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 128
    n_heads: int = 8
    e_layers: int = 2
    d_ff: int = 256
    dropout: float = 0.1
    window_size: list[int] = [4, 4]
    inner_size: int = 5
    embed: str = "timeF"
    freq: str = "h"
