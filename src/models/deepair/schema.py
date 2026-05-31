"""Parameter schema for the DeepAir model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated DeepAir parameters supplied via ``model.params``."""

    enc_in: int
    cov_dim: int = 2
    hid_dim: int = 64
