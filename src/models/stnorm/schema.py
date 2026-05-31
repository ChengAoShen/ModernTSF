"""Parameter schema for the STNorm model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated STNorm parameters supplied via ``model.params``."""

    enc_in: int
    cov_dim: int | None = None
    channels: int = 16
    kernel_size: int = 2
    blocks: int = 8
    layers: int = 2
