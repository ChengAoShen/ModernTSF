"""Parameter schema for the STOP model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated STOP parameters supplied via ``model.params``."""

    enc_in: int
    model_dim: int = 16
    prompt_dim: int = 16
    num_layer: int = 2
    hid_dim: int = 64
    tod_size: int = 24
    kernel_size: int = 3
    core: int = 4
    head: int = 4
