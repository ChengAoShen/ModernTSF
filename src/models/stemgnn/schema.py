"""Parameter schema for the StemGNN model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated StemGNN parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset
    (not declared here). ``enc_in`` (= number of nodes ``N``) is required.
    """

    enc_in: int
    input_dim: int = 3
    multi_layer: int = 3
    dropout_rate: float = 0.5
    leaky_rate: float = 0.2
