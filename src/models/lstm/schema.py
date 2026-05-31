"""Parameter schema for the LSTM baseline."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int = 207
    init_dim: int = 32
    hid_dim: int = 64
    end_dim: int = 128
    layer: int = 2
    dropout: float = 0.1
    cov_dim: int = 2
