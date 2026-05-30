"""Parameter schema for the MAGE model."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated MAGE parameters supplied via ``model.params``."""

    enc_in: int
    model_dim: int = 64
    recur_num: int = 8
    blocknum: int = 3
    topk: int = 2
    node_dim: int = 16
    tod_size: int = 24
