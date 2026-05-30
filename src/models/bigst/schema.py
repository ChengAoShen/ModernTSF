"""Parameter schema for the BigST model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated BigST parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset
    and are not declared here.
    """

    enc_in: int  # number of nodes N (required)
    input_dim: int = 3  # 1 value + calendar covariates [tod, dow]
    hid_dim: int = 16
    node_dim: int = 8
    time_dim: int = 8
    tod_size: int = 24
    dow_size: int = 7
    tau: float = 1.0
    random_feature_dim: int = 16
    dropout: float = 0.1
    use_residual: bool = True
    use_bn: bool = True
