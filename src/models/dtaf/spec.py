"""Model specification for DTAF."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec
from models.dtaf.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 32
    e_layers: int = 1
    patch_len: int = 16
    stride: int = 8
    heads: int = 2
    dropout: float = 0.1
    moving_avg: int = 25
    expert_num: int = 2
    kan_div: int = 4
    k: int = 1
    aggregated_norm: int = 1


def build_model(cfg, params):
    """Construct DTAF from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, label_len=cfg.task.label_len, features=cfg.task.features, enc_in=params['enc_in'], d_model=params.get('d_model', 32), e_layers=params.get('e_layers', 1), patch_len=params.get('patch_len', 16), stride=params.get('stride', 8), heads=params.get('heads', 2), dropout=params.get('dropout', 0.1), moving_avg=params.get('moving_avg', 25), expert_num=params.get('expert_num', 2), kan_div=params.get('kan_div', 4), k=params.get('k', 1), aggregated_norm=params.get('aggregated_norm', 1))
    )


SPEC = ModelSpec(
    name='DTAF',
    module='models.dtaf',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/DTAF.toml',
    model_card='src/models/dtaf/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('autoformer_encdec', 'embed'),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
