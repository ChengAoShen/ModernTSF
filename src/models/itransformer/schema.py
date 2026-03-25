from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    freq: str = "h"
    embed: str = "timeF"
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 2
    d_ff: int = 2048
    factor: int = 1
    dropout: float = 0.1
    activation: str = "gelu"
    output_attention: bool = False
    use_norm: bool = True
    class_strategy: str = "projection"
