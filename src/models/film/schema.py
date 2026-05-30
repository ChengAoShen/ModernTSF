from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    e_layers: int = 2
    ratio: float = 0.5
    multiscale: list[int] = [1, 2, 4]
    window_size: list[int] = [256]
