"""Parameter schema for the DSTAGNN model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated DSTAGNN parameters supplied via ``model.params``."""

    enc_in: int
    d_model: int = 64
    d_k: int = 8
    d_v: int = 8
    n_heads: int = 4
    cov_dim: int = 2
