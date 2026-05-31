"""Parameter schema for the PCDCNet model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated PCDCNet parameters supplied via ``model.params``."""

    enc_in: int
    d_model: int = 64
    dropout: float = 0.1
    cov_dim: int | None = None
