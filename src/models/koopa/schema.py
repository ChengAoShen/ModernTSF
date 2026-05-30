from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    seg_len: int | None = None
    dynamic_dim: int = 128
    hidden_dim: int = 64
    hidden_layers: int = 2
    num_blocks: int = 3
    multistep: bool = False
    alpha: float = 0.2
