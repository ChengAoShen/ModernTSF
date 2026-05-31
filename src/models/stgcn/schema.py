"""Parameter schema for the STGCN model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated STGCN parameters supplied via ``model.params``."""

    enc_in: int
    cov_dim: int = 2
    Ks: int = 3
    Kt: int = 3
    blocks: list | None = None
    drop_prob: float = 0.0
