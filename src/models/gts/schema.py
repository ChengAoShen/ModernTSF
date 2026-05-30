"""Parameter schema for the GTS model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated GTS parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset and
    need not be declared in the TOML. ``enc_in`` (= number of nodes ``N``) is the
    only required field and serves as the fallback for ``num_nodes``.
    """

    enc_in: int
    input_dim: int = 3
    rnn_units: int = 16
    num_rnn_layers: int = 1
    max_diffusion_step: int = 2
    embedding_dim: int = 16
    node_feats_len: int = 40
    k: int = 3
    temp: float = 0.5
