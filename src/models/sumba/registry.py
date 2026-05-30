"""Model registration for Sumba."""

from benchmark.registry import MODEL_REGISTRY
from models.sumba.model import Model
from models.sumba.schema import ModelParameterConfig


def register() -> None:
    """Register Sumba model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "Sumba",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            label_len=cfg.task.label_len,
            features=cfg.task.features,
            enc_in=params["enc_in"],
            input_dim=params.get("input_dim", 1),
            output_dim=params.get("output_dim", 1),
            residual_channels=params.get("residual_channels", 16),
            conv_channels=params.get("conv_channels", 16),
            skip_channels=params.get("skip_channels", 32),
            end_channels=params.get("end_channels", 64),
            dimension=params.get("dimension", 16),
            M=params.get("M", 4),
            LowRank=params.get("LowRank", 8),
            D=params.get("D", 16),
            gcn_depth=params.get("gcn_depth", 2),
            sumba_layers=params.get("sumba_layers", 2),
            layers=params.get("layers", 2),
            dilation_exponential=params.get("dilation_exponential", 1),
            kernel_set=tuple(params.get("kernel_set", (2, 3, 6, 7))),
            propalpha=params.get("propalpha", 0.05),
            dropout=params.get("dropout", 0.3),
            layer_norm_affline=bool(params.get("layer_norm_affline", True)),
            mark_dim=params.get("mark_dim", 6),
        ),
        ModelParameterConfig,
    )
