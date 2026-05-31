"""Parameter schema for the STTN model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated STTN parameters supplied via ``model.params``."""

    enc_in: int
    cov_dim: int = 2
    d_model: int = 64
    n_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.1
    adj_type: str = "doubletransition"
