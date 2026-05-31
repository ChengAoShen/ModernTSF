"""Parameter schema for the PM25_GNN model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated PM25_GNN parameters supplied via ``model.params``."""

    enc_in: int
    cov_dim: int = 2
    hid_dim: int = 64
