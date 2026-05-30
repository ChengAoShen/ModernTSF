from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 256
    d_ff: int = 64
    e_layers: int = 2
    fac_T: bool = False
    fac_C: bool = False
    sampling: int = 2
    norm: bool = True
    individual: bool = False
    rev: bool = True
