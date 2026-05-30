"""Parameter schema for the MoFo model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated MoFo parameters supplied via ``model.params``."""

    enc_in: int
    d_model: int = 64
    periodic: int = 24
    head: int = 4
    d_layers: int = 1
    bias: int = 1
    cias: int = 1
