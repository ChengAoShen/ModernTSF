from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    c_out: int = 0  # 0 -> defaults to enc_in in the registry factory
    d_model: int = 64
    n_heads: int = 4
    d_layers: int = 1
    dropout: float = 0.05
    embed: str = "timeF"
    freq: str = "h"
    conv_kernel: list[int] = [12, 16]
