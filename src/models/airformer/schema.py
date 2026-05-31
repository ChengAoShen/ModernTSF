"""Parameter schema for the AirFormer model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated AirFormer parameters supplied via ``model.params``."""

    enc_in: int
    d_model: int = 32
    nhead: int = 2
    num_encoder_layers: int = 4
    dropout: float = 0.3
    cov_dim: int = 2
