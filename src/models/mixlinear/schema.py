from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    period_len: int = 24
    com_len: int = 4
    lpf: int = 1
    alpha: float = 0.5
