"""Parameter schema for the STID model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated STID parameters supplied via ``model.params``."""

    enc_in: int
    cov_dim: int = 2
    hid_dim: int = 64
    num_layers: int = 3
    time_of_day_size: int = 24
    day_of_week_size: int = 7
