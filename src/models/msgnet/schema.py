from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    c_out: int | None = None
    d_model: int = 128
    d_ff: int = 256
    e_layers: int = 2
    n_heads: int = 8
    top_k: int = 5
    dropout: float = 0.1
    conv_channel: int = 32
    skip_channel: int = 32
    gcn_depth: int = 2
    propalpha: float = 0.3
    node_dim: int = 10
    individual: bool = False
    embed: str = "timeF"
    freq: str = "h"
