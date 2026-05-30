"""Parameter schema for the AirCade model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated AirCade parameters supplied via ``model.params``."""

    enc_in: int
    cov_dim: int | None = None
    input_embedding_dim: int = 16
    adaptive_embedding_dim: int = 24
    feed_forward_dim: int = 64
    num_heads: int = 4
    num_layers: int = 1
    node_embed_dim: int = 10
