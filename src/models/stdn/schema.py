"""Parameter schema for the STDN model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated STDN parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset
    (not declared in TOML). ``enc_in`` (= ``N``) is the required node count and
    is used as the ``num_nodes`` fallback.
    """

    enc_in: int
    input_dim: int = 3
    time_slice_size: int = 60
    K: int = 4
    d: int = 8
    L: int = 1
    order: int = 2
    reference: int = 4
    out_channels: int = 1
