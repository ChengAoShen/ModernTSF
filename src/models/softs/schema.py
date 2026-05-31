"""Parameter schema for the SOFTS model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated SOFTS parameters supplied via ``model.params``."""

    enc_in: int
    patch_len: int = 24
    d_model: int = 128
    dropout: float = 0.1
    n_heads: int = 4
    e_layers: int = 2
    d_ff: int = 256
