from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    dropout: float = 0.1
    seg_len: int = 24
