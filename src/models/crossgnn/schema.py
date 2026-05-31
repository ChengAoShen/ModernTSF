"""Parameter schema for the CrossGNN model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated CrossGNN parameters supplied via ``model.params``."""

    enc_in: int
    cov_dim: int = 2
    seg_len: int = 6
    d_model: int = 128
    d_ff: int = 256
    n_heads: int = 4
    e_layers: int = 3
    dropout: float = 0.1
