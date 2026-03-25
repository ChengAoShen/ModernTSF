from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    freq: str = "h"
    d_model: int = 512
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 2048
    c_out: int = 7
    dropout: float = 0.1
    bias: bool = True
    feature_encode_dim: int = 2
