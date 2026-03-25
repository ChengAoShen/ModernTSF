"""Model registration for CMoS."""

from benchmark.registry import MODEL_REGISTRY
from models.cmos.model import Model
from models.cmos.schema import ModelParameterConfig


def register() -> None:
    """Register CMoS model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "CMoS",
        lambda cfg, params: Model(
            c_in=params["enc_in"],
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            seg_size=params.get("seg_size", 4),
            num_map=params.get("num_map", 3),
            kernel_size=params.get("kernel_size", 3),
            conv_stride=params.get("conv_stride", 1),
            topk=params.get("topk", 3),
            dropout=params.get("dropout", 0.1),
        ),
        ModelParameterConfig,
    )
