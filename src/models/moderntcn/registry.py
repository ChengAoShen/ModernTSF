"""Model registration for ModernTCN."""

from benchmark.registry import MODEL_REGISTRY
from models.moderntcn.model import Model
from models.moderntcn.schema import ModelParameterConfig


def register() -> None:
    """Register ModernTCN model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "ModernTCN",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            label_len=cfg.task.label_len,
            features=cfg.task.features,
            enc_in=params["enc_in"],
            ffn_ratio=params.get("ffn_ratio", 1),
            num_blocks=params.get("num_blocks", [1]),
            large_size=params.get("large_size", [13]),
            small_size=params.get("small_size", [5]),
            dims=params.get("dims", [32]),
            dw_dims=params.get("dw_dims", [32]),
            patch_size=params.get("patch_size", 16),
            patch_stride=params.get("patch_stride", 16),
            stem_ratio=params.get("stem_ratio", 6),
            downsample_ratio=params.get("downsample_ratio", 2),
            small_kernel_merged=bool(params.get("small_kernel_merged", False)),
            dropout=params.get("dropout", 0.1),
            head_dropout=params.get("head_dropout", 0.1),
            use_multi_scale=bool(params.get("use_multi_scale", True)),
            revin=bool(params.get("revin", True)),
            affine=bool(params.get("affine", True)),
            subtract_last=bool(params.get("subtract_last", False)),
            individual=bool(params.get("individual", False)),
            decomposition=bool(params.get("decomposition", False)),
            kernel_size=params.get("kernel_size", 25),
        ),
        ModelParameterConfig,
    )
