"""Model registration for NHiTS."""

from benchmark.registry import MODEL_REGISTRY
from models.nhits.model import Model
from models.nhits.schema import ModelParameterConfig


def register() -> None:
    """Register NHiTS model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "NHiTS",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            label_len=cfg.task.label_len,
            features=cfg.task.features,
            enc_in=params["enc_in"],
            stack_types=params.get("stack_types", ["identity", "identity", "identity"]),
            n_blocks=params.get("n_blocks", [1, 1, 1]),
            mlp_units=params.get("mlp_units", [[256, 256]]),
            n_pool_kernel_size=params.get("n_pool_kernel_size", [2, 2, 1]),
            n_freq_downsample=params.get("n_freq_downsample", [4, 2, 1]),
            pooling_mode=params.get("pooling_mode", "MaxPool1d"),
            interpolation_mode=params.get("interpolation_mode", "linear"),
            dropout=params.get("dropout", 0.0),
            activation=params.get("activation", "ReLU"),
            use_norm=bool(params.get("use_norm", True)),
        ),
        ModelParameterConfig,
    )
