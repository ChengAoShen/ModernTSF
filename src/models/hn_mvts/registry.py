"""Model registration for HN_MVTS."""

from benchmark.registry import MODEL_REGISTRY
from models.hn_mvts.model import Model
from models.hn_mvts.schema import ModelParameterConfig


def register() -> None:
    """Register HN_MVTS model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "HN_MVTS",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            d_model=params.get("d_model", 64),
            dropout=params.get("dropout", 0.1),
            period=params.get("period", 24),
            num_prompts=params.get("num_prompts", 4),
            use_revin=bool(params.get("use_revin", True)),
        ),
        ModelParameterConfig,
    )
