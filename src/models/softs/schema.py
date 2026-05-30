from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 128
    d_core: int = 64
    d_ff: int = 256
    e_layers: int = 2
    dropout: float = 0.1
    activation: str = "gelu"
    use_norm: bool = True
