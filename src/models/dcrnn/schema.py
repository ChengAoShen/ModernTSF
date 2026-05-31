"""Parameter schema for the DCRNN model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated DCRNN parameters supplied via ``model.params``."""

    enc_in: int
    cov_dim: int = 2
    n_filters: int = 64
    max_diffusion_step: int = 2
    filter_type: str = "doubletransition"
    num_rnn_layers: int = 2
