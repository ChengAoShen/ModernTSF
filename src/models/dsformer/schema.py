from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    num_layer: int = 1
    dropout: float = 0.2
    muti_head: int = 4
    num_samp: int = 3
