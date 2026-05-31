from pydantic import BaseModel
from typing import List


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 32
    d_ff: int = 64
    layer_nums: int = 3
    k: int = 2
    patch_size_list: List[int] = [3, 5, 7]
    num_experts_list: List[int] = [4, 4, 4]
    revin: bool = True
    residual_connection: int = 1
    batch_norm: bool = False
