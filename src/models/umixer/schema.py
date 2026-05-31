from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    patch_len: int = 24
    stride: int = 24
    d_model: int = 128
    dropout: float = 0.1
    e_layers: int = 2
    d_layers: int = 1
