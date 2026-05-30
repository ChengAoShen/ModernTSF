"""Parameter schema for the MTGNN model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated MTGNN parameters supplied via ``model.params``.

    ``enc_in`` (= number of nodes ``N``) is required. ``num_nodes`` and
    ``adj_mx`` are injected by the runner from the dataset and need not be
    declared in the TOML.
    """

    enc_in: int
    input_dim: int = 3
    gcn_depth: int = 2
    subgraph_size: int = 20
    node_dim: int = 40
    conv_channels: int = 16
    residual_channels: int = 16
    skip_channels: int = 32
    end_channels: int = 64
    layers: int = 3
    dropout: float = 0.3
    propalpha: float = 0.05
    tanhalpha: float = 3.0
    dilation_exponential: int = 1
    build_adj: bool = True
