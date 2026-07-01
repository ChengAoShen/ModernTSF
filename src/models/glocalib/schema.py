from typing import Literal

from pydantic import BaseModel, Field


class ModelParameterConfig(BaseModel):
    """Parameters for the Glocal-IB forecasting model."""

    enc_in: int = Field(gt=0)
    d_model: int = Field(default=64, gt=0)
    # Alignment regularizer weight and augmentation strength.
    align_weight: float = Field(default=0.5, ge=0.0)
    mask_ratio: float = Field(default=0.25, ge=0.0, lt=1.0)
    # "cos_align" (robust) or "contrastive" (InfoNCE over time steps).
    align_loss_type: Literal["cos_align", "contrastive"] = "cos_align"
