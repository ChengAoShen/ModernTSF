"""Model specification for TiDE."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec
from models.tide.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    d_model: int = 512
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 2048
    decoder_output_dim: int = 7
    time_feat_dim: int = 6
    dropout: float = 0.1
    bias: bool = True
    feature_encode_dim: int = 2


def build_model(cfg, params):
    """Construct TiDE from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        d_model=params.get('d_model', 512),
        e_layers=params.get('e_layers', 2),
        d_layers=params.get('d_layers', 1),
        d_ff=params.get('d_ff', 2048),
        decoder_output_dim=params.get('decoder_output_dim', 7),
        time_feat_dim=params.get('time_feat_dim', 6),
        dropout=params.get('dropout', 0.1),
        bias=bool(params.get('bias', True)),
        feature_encode_dim=params.get('feature_encode_dim', 2),
    )


SPEC = ModelSpec(
    name='TiDE',
    module='models.tide',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/TiDE.toml',
    model_card='src/models/tide/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
