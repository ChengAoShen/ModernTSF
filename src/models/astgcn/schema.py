"""Parameter schema for the ASTGCN model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated ASTGCN parameters supplied via ``model.params``."""

    enc_in: int
    cov_dim: int = 2
    nb_block: int = 2
    K: int = 3
    nb_chev_filter: int = 64
    nb_time_filter: int = 64
