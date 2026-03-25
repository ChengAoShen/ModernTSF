from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    individual: bool = False
    cut_freq: int = 24
