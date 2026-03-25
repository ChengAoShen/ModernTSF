"""Model registration for FITS."""

from benchmark.registry import MODEL_REGISTRY
from models.fits.model import Model
from models.fits.schema import ModelParameterConfig


def register() -> None:
    """Register FITS model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "FITS",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            individual=bool(params.get("individual", False)),
            cut_freq=params.get("cut_freq", 24),
        ),
        ModelParameterConfig,
    )
