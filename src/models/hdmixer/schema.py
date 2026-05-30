from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 128
    d_ff: int = 256
    e_layers: int = 3
    patch_len: int = 16
    stride: int = 8
    dropout: float = 0.1
    head_dropout: float = 0.0
    activation: str = "gelu"
    individual: bool = False
    revin: bool = True
    affine: bool = True
    subtract_last: bool = False
    deform_range: float = 0.25
    mix_time: bool = True
    mix_variable: bool = True
    mix_channel: bool = True
