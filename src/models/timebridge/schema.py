from typing import Optional

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    period: int = 24
    num_p: Optional[int] = None
    ia_layers: int = 2
    pd_layers: int = 1
    ca_layers: int = 2
    stable_len: int = 3
    d_model: int = 16
    n_heads: int = 4
    d_ff: int = 128
    attn_dropout: float = 0.15
    dropout: float = 0.0
    activation: str = "gelu"
    revin: bool = True
    time_feat_dim: int = 6
