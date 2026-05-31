"""Parameter schema for the RPMixer model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated RPMixer parameters supplied via ``model.params``."""

    enc_in: int
    cov_dim: int | None = None
    IE_dim: int = 32
    dropout: float = 0.3
    num_head: int = 2
