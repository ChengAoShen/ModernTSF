"""Parameter schema for the AirPhyNet model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated AirPhyNet parameters supplied via ``model.params``."""

    enc_in: int
    latent_dim: int = 4
    rnn_units: int = 64
    ode_method: str = "dopri5"
    cov_dim: int = 2
