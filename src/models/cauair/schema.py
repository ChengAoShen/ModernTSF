"""Parameter schema for the CauAir model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated CauAir parameters supplied via ``model.params``."""

    enc_in: int
    cov_dim: int | None = None
    dim: int = 64
    rank: int = 8
    head: int = 4
