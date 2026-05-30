"""Parameter schema for the DGCRN model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated DGCRN parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset
    (see ``run_one.py``) and are therefore not declared here.
    """

    enc_in: int
    input_dim: int = 2
    gcn_depth: int = 1
    rnn_size: int = 16
    node_dim: int = 8
    hyper_gnn_dim: int = 8
    middle_dim: int = 2
    subgraph_size: int = 20
    tanhalpha: float = 3.0
    dropout: float = 0.3
