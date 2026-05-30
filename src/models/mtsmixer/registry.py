"""Model registration for MTSMixer."""

from benchmark.registry import MODEL_REGISTRY
from models.mtsmixer.model import Model
from models.mtsmixer.schema import ModelParameterConfig


def register() -> None:
    """Register MTSMixer model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "MTSMixer",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            features=cfg.task.features,
            enc_in=params["enc_in"],
            d_model=params.get("d_model", 256),
            d_ff=params.get("d_ff", 64),
            e_layers=params.get("e_layers", 2),
            fac_T=bool(params.get("fac_T", False)),
            fac_C=bool(params.get("fac_C", False)),
            sampling=params.get("sampling", 2),
            norm=bool(params.get("norm", True)),
            individual=bool(params.get("individual", False)),
            rev=bool(params.get("rev", True)),
        ),
        ModelParameterConfig,
    )
