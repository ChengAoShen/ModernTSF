from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_layers: int = 1
    dropout: float = 0.0
