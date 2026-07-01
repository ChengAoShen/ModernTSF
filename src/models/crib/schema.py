from typing import Literal

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    """Parameters for the CRIB model.

    Note: ``patch_len`` must divide ``task.seq_len``; ``model_dim`` must be
    divisible by ``heads_num``.
    """

    enc_in: int = Field(gt=0)
    patch_len: int = Field(default=8, gt=0)
    model_dim: int = Field(default=32, gt=0)
    heads_num: int = Field(default=4, gt=0)
    enc_num: int = Field(default=3, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    activation: Literal["relu", "gelu"] = "relu"
    # Consistency (MSE) and IB (KL) regularizer weights (upstream 1.0 / 1e-6).
    consis_weight: float = Field(default=1.0, ge=0.0)
    kl_weight: float = Field(default=1e-6, ge=0.0)
