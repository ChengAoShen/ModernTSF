"""Model registration for SparseTSF."""

from benchmark.registry import MODEL_REGISTRY
from models.sparsetsf.model import Model
from models.sparsetsf.schema import ModelParameterConfig


def register() -> None:
    """Register SparseTSF model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "SparseTSF",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            period=params.get("period", 24),
            d_model=params.get("d_model", 64),
            model_type=params.get("model_type", "linear"),
        ),
        ModelParameterConfig,
    )
