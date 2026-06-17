from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    eps: float = 1e-6
