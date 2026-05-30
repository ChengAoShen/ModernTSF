"""Parameter schema for the D2STGNN model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated D2STGNN parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset
    (see ``run_one``) and are therefore not declared here.
    """

    enc_in: int
    input_dim: int = 3
    num_feat: int = 1
    num_hidden: int = 16
    node_hidden: int = 8
    time_emb_dim: int = 8
    k_s: int = 2
    k_t: int = 3
    gap: int = 1
    num_layers: int = 2
    dropout: float = 0.1
    time_in_day_size: int = 288
    day_in_week_size: int = 7
    forecast_dim: int = 64
    output_hidden: int = 128
