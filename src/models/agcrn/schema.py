"""Parameter schema for the AGCRN model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated AGCRN parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset and
    are therefore not declared here. ``enc_in`` (= number of nodes ``N``) is the
    only required field; the remaining defaults are kept modest for a fast
    spatiotemporal smoke.
    """

    enc_in: int
    input_dim: int = 3
    rnn_units: int = 32
    embed_dim: int = 8
    num_layers: int = 1
    cheb_k: int = 2
    output_dim: int = 1
