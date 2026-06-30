from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Parameters for the Glocal-IB forecasting model."""

    enc_in: int
    d_model: int = 64
    # Alignment regularizer weight and augmentation strength.
    align_weight: float = 0.5
    mask_ratio: float = 0.25
    # "cos_align" (robust) or "contrastive" (InfoNCE over time steps).
    align_loss_type: str = "cos_align"
