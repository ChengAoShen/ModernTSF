from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    c_out: int | None = None
    d_model: int = 64
    e_layers: int = 2
    patch_len: int = 16
    stride: int = 8
    dropout: float = 0.1
