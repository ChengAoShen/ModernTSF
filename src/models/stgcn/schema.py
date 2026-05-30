"""Parameter schema for the STGCN model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated STGCN parameters supplied via ``model.params``.

    ``enc_in`` (the number of spatial nodes ``N``) is required. ``num_nodes``
    and ``adj_mx`` are injected by the runner from the dataset and are NOT
    declared in the TOML.
    """

    enc_in: int
    input_dim: int = 3
    Kt: int = 3
    Ks: int = 3
    hidden_dim: int = 64
    bottleneck_dim: int = 16
    out_hidden_dim: int = 128
    act_func: str = "glu"
    graph_conv_type: str = "cheb_graph_conv"
    bias: bool = True
    droprate: float = 0.5
