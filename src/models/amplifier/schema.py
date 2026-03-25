from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    hidden_size: int = 128
    sci: bool = False
