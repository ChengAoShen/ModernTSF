"""Parameter schema for the STNorm model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated STNorm parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset
    and are therefore not declared here.
    """

    enc_in: int
    input_dim: int = 3
    channels: int = 16
    kernel_size: int = 2
    blocks: int = 2
    layers: int = 2
    tnorm_bool: bool = True
    snorm_bool: bool = True
