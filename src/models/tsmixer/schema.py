from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    e_layers: int = 2
    dropout: float = 0.1
