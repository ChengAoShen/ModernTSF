from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    patch_len: int = 16
    stride: int = 8
    padding_patch: str = "end"
    ma_type: str = "ema"
    alpha: float = 0.3
    beta: float = 0.3
    revin: bool = True
