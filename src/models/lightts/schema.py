from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    hid_dim: int = 128
    dropout: float = 0.0
    chunk_size: int = 40
    c_dim: int = 40
