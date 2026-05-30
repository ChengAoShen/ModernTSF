"""Parameter schema for the DFDGCN model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated DFDGCN parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset
    (see ``run_one.py``) and are therefore not declared here.
    """

    enc_in: int
    input_dim: int = 3
    dropout: float = 0.3
    residual_channels: int = 16
    dilation_channels: int = 16
    skip_channels: int = 64
    end_channels: int = 128
    kernel_size: int = 2
    blocks: int = 2
    layers: int = 2
    a: float = 1.0
    fft_emb: int = 10
    identity_emb: int = 10
    hidden_emb: int = 30
    subgraph: int = 20
