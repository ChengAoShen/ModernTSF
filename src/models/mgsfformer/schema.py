"""Parameter schema for the MGSFformer model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated MGSFformer parameters supplied via ``model.params``."""

    enc_in: int
    IE_dim: int = 32
    dropout: float = 0.3
    num_head: int = 2
