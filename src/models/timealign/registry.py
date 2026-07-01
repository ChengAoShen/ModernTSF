"""Model registration for TimeAlign."""

from benchmark.registry import MODEL_REGISTRY
from models.timealign.model import Model
from models.timealign.schema import ModelParameterConfig


def register() -> None:
    """Register TimeAlign model factory and parameter schema."""
    MODEL_REGISTRY.register(
        "TimeAlign",
        lambda cfg, params: Model(
            seq_len=cfg.task.seq_len,
            pred_len=cfg.task.pred_len,
            enc_in=params["enc_in"],
            patch_num=params.get("patch_num", 4),
            d_model=params.get("d_model", 32),
            d_ff=params.get("d_ff", 32),
            e_layers=params.get("e_layers", 2),
            dropout=params.get("dropout", 0.1),
            pos=bool(params.get("pos", True)),
            layer_norm=bool(params.get("layer_norm", True)),
            loc=bool(params.get("loc", True)),
            glo=bool(params.get("glo", True)),
            local_margin=params.get("local_margin", 0.0),
            global_margin=params.get("global_margin", 0.0),
            w_recon=params.get("w_recon", 1.0),
            w_align=params.get("w_align", 0.1),
        ),
        ModelParameterConfig,
    )
