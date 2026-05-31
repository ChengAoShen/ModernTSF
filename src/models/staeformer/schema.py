"""Parameter schema for the STAEformer model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated STAEformer parameters supplied via ``model.params``."""

    enc_in: int
    cov_dim: int | None = None
    input_embedding_dim: int = 24
    tod_embedding_dim: int = 24
    dow_embedding_dim: int = 24
    adaptive_embedding_dim: int = 56
    feed_forward_dim: int = 128
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    steps_per_day: int = 24
