"""Model registration for Pathformer."""

from benchmark.registry import MODEL_REGISTRY
from models.pathformer.model import Model
from models.pathformer.schema import ModelParameterConfig


def register() -> None:
    """Register Pathformer model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "Pathformer",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            features=cfg.task.features,
            enc_in=params["enc_in"],
            layer_nums=params.get("layer_nums", 2),
            k=params.get("k", 2),
            num_experts=params.get("num_experts", 4),
            patch_size_list=params.get(
                "patch_size_list", [16, 12, 8, 6, 16, 12, 8, 6]
            ),
            d_model=params.get("d_model", 16),
            d_ff=params.get("d_ff", 64),
            residual_connection=params.get("residual_connection", 1),
            revin=bool(params.get("revin", True)),
            batch_norm=bool(params.get("batch_norm", False)),
        ),
        ModelParameterConfig,
    )
