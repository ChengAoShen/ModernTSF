from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 128
    d_state: int = 16
    d_ff: int = 128
    e_layers: int = 2
    d_conv: int = 2
    expand: int = 1
    dropout: float = 0.1
    activation: str = "gelu"
    use_norm: bool = True
    embed: str = "timeF"
    freq: str = "h"
