"""Model registration for PatchMLP."""

from benchmark.registry import MODEL_REGISTRY
from models.patchmlp.model import Model
from models.patchmlp.schema import ModelParameterConfig


def register() -> None:
    """Register PatchMLP model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "PatchMLP",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            d_model=params.get("d_model", 1024),
            e_layers=params.get("e_layers", 1),
            use_norm=bool(params.get("use_norm", True)),
            moving_avg=params.get("moving_avg", 13),
            patch_len=params.get("patch_len"),
        ),
        ModelParameterConfig,
    )
