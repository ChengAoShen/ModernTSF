"""Parameter schema for the D2STGNN model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated D2STGNN parameters supplied via ``model.params``."""

    enc_in: int
    d_model: int = 64
    num_layers: int = 4
    dropout: float = 0.1
    cov_dim: int = 2
