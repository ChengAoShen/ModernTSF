"""Parameter schema for the PHAT model."""

from typing import Optional

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated PHAT parameters supplied via ``model.params``."""

    enc_in: int
    d_model: int = 64
    n_heads: int = 8
    d_layers: int = 1
    attn_dropout: float = 0.1
    ffn_dropout: float = 0.1
    ffn_expand_ratio: float = 2.66667
    period_topk: int = 1
    period_list: Optional[list[int]] = None
    ci: int = 1
    output_base_pred: int = 0
