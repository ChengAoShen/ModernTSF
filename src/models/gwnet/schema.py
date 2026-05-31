"""Parameter schema for the GWNet model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated GWNet parameters supplied via ``model.params``."""

    enc_in: int
    cov_dim: int = 2
    residual_channels: int = 32
    dilation_channels: int = 32
    skip_channels: int = 64
    end_channels: int = 128
    dropout: float = 0.3
    blocks: int = 8
    layers: int = 2
    adp_adj: bool = True
    adj_type: str = "doubletransition"
