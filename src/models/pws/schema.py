from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    period: int = 24
    patch_size: int = 6
    revin: bool = True
    affine: bool = False
    subtract_last: bool = False
    analysis_act: str = "relu"
    analysis_hidden: str = "512,256"
