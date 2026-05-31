from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 128
    n_heads: int = 16
    d_ff: int = 256
    n_layers: int = 3
    dropout: float = 0.1
    stride: int = 24
    QAM_start: float = 0.1
    QAM_end: float = 0.5
