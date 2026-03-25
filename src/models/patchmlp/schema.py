from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 1024
    e_layers: int = 1
    use_norm: bool = True
    moving_avg: int = 13
    patch_len: list[int] = [48, 24, 12, 6]
