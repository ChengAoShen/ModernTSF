"""Model registration for Pathformer."""

from benchmark.registry import MODEL_REGISTRY
from models.pathformer.model import Model
from models.pathformer.schema import ModelParameterConfig


def register() -> None:
    """Register the Pathformer model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "Pathformer",
        lambda cfg, params, **kw: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            d_model=params.get("d_model", 32),
            d_ff=params.get("d_ff", 64),
            layer_nums=params.get("layer_nums", 3),
            k=params.get("k", 2),
            patch_size_list=params.get("patch_size_list", [3, 5, 7]),
            num_experts_list=params.get("num_experts_list", [4, 4, 4]),
            revin=params.get("revin", True),
            residual_connection=params.get("residual_connection", 1),
            batch_norm=params.get("batch_norm", False),
        ),
        ModelParameterConfig,
    )
