"""Parameter schema for the STAEformer model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated STAEformer parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset
    (see ``src/benchmark/runner/run_one.py``) and need not be declared in the
    TOML. ``enc_in`` (the node count ``N``) is the only required field and is
    used as the fallback node count.
    """

    enc_in: int
    input_dim: int = 3
    steps_per_day: int = 24
    input_embedding_dim: int = 24
    tod_embedding_dim: int = 24
    dow_embedding_dim: int = 24
    spatial_embedding_dim: int = 0
    adaptive_embedding_dim: int = 80
    feed_forward_dim: int = 256
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.1
    use_mixed_proj: bool = True
