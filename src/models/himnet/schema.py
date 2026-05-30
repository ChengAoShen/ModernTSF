"""Parameter schema for the HimNet model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated HimNet parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset and
    need not be declared in TOML. ``enc_in`` (= number of nodes ``N``) is the
    required channel count; the remaining fields carry modest defaults for a
    fast smoke run.
    """

    enc_in: int
    input_dim: int = 3
    output_dim: int = 1
    hidden_dim: int = 32
    num_layers: int = 1
    cheb_k: int = 2
    node_embedding_dim: int = 8
    st_embedding_dim: int = 8
    tod_embedding_dim: int = 8
    dow_embedding_dim: int = 8
    steps_per_day: int = 288
    use_teacher_forcing: bool = True
