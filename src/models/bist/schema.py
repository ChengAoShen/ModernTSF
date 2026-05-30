"""Parameter schema for the BiST model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated BiST parameters supplied via ``model.params``."""

    enc_in: int
    model_dim: int = 32
    prompt_dim: int = 32
    num_layer: int = 2
    hid_dim: int = 64
    tod_size: int = 24
    kernel_size: int = 3
    rp_layer: int = 1
    adaptive_adj_dim: int = 10
    core: int = 0
