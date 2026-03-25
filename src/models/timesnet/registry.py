"""Model registration for TimesNet."""

from benchmark.registry import MODEL_REGISTRY
from models.timesnet.model import Model
from models.timesnet.schema import ModelParameterConfig


def register() -> None:
    """Register TimesNet model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "TimesNet",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            label_len=cfg.task.label_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            c_out=params["c_out"],
            d_model=params.get("d_model", 512),
            e_layers=params.get("e_layers", 2),
            d_ff=params.get("d_ff", 2048),
            freq=params.get("freq", "h"),
            dropout=params.get("dropout", 0.1),
            embed=params.get("embed", "timeF"),
            top_k=params.get("top_k", 5),
            num_kernels=params.get("num_kernels", 6),
            num_class=params.get("num_class", 7),
            task_name=params.get("task_name", "long_term_forecast"),
        ),
        ModelParameterConfig,
    )
