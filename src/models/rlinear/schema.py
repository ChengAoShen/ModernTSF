from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    individual: bool = False
    affine: bool = False
    subtract_last: bool = False
