from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    seg_size: int = 4
    num_map: int = 3
    kernel_size: int = 3
    conv_stride: int = 1
    topk: int = 3
    dropout: float = 0.1
