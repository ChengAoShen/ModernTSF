from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    d_ff: int = 128
    n_heads: int = 4
    e_layers: int = 2
    patch_len: int = 16
    dropout: float = 0.1
    alpha: float = 0.1
    top_p: float = 0.5
    pos: bool = True
