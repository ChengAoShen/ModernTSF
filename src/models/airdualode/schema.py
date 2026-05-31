"""Parameter schema for the AirDualODE model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated AirDualODE parameters supplied via ``model.params``."""

    enc_in: int
    phy_latent_dim: int = 16
    unk_latent_dim: int = 16
    gcn_hidden_dim: int = 32
    n_heads: int = 4
    ode_method: str = "euler"
    cov_dim: int = 2
