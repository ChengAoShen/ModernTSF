from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    num_layer: int = 1
    muti_head: int = 2
    num_samp: int = 2
    dropout: float = 0.15
    if_node: bool = True
