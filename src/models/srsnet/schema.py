from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 128
    patch_len: int = 24
    stride: int = 24
    hidden_size: int = 64
    dropout: float = 0.2
    head_dropout: float = 0.1
    alpha: float = 2.0
    pos: bool = True
    head_mode: str = "linear"
    affine: bool = True
    subtract_last: bool = False
