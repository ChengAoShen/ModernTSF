"""Parameter schema for the DGCRN model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated DGCRN parameters supplied via ``model.params``."""

    enc_in: int
    cov_dim: int = 2
    gcn_depth: int = 2
    dropout: float = 0.3
    subgraph_size: int = 20
    node_dim: int = 40
    rnn_size: int = 64
    adj_type: str = "doubletransition"
