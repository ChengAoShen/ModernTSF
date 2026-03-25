from typing import Optional

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    dec_in: Optional[int] = None
    patch_len: int = 16
    d_model: int = 32
    d_ff: int = 2048
    alpha: float = 1.0
    beta: float = 0.5
