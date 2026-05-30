"""Parameter schema for the STPGNN model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated STPGNN parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset and
    must not be declared in the TOML. ``enc_in`` (= number of nodes ``N``) is the
    required spatial dimension; the remaining fields are kept small so the smoke
    run stays fast.
    """

    enc_in: int
    input_dim: int = 3
    dropout: float = 0.1
    topk: int = 4
    residual_channels: int = 16
    dilation_channels: int = 16
    end_channels: int = 64
    kernel_size: int = 2
    blocks: int = 2
    layers: int = 2
    days: int = 7
    time_of_day_size: int = 24
    dims: int = 16
    order: int = 2
    normalization: str = "batch"
