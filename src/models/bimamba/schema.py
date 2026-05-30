from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    c_out: int | None = None
    d_model: int = 128
    d_state: int = 16
    e_layers: int = 2
    expand: int = 2
    d_conv: int = 4
    dropout: float = 0.1
    share_ffn: bool = False
    share_norm: bool = False
    embed: str = "timeF"
    freq: str = "h"
