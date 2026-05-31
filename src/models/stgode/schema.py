"""Parameter schema for the STGODE model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated STGODE parameters supplied via ``model.params``."""

    enc_in: int
    cov_dim: int = 2
    num_layers: int = 3
