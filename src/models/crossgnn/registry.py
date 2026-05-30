"""Model registration for CrossGNN."""

from benchmark.registry import MODEL_REGISTRY
from models.crossgnn.model import Model
from models.crossgnn.schema import ModelParameterConfig


def register() -> None:
    """Register CrossGNN model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "CrossGNN",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            e_layers=params.get("e_layers", 2),
            anti_ood=bool(params.get("anti_ood", True)),
            tk=params.get("tk", 10),
            scale_number=params.get("scale_number", 4),
            use_tgcn=bool(params.get("use_tgcn", True)),
            use_ngcn=bool(params.get("use_ngcn", True)),
            individual=bool(params.get("individual", False)),
            dropout=params.get("dropout", 0.1),
            tvechidden=params.get("tvechidden", 8),
            nvechidden=params.get("nvechidden", 8),
            hidden=params.get("hidden", 16),
        ),
        ModelParameterConfig,
    )
