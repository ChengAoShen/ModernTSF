"""Model specification for MixLinear."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec
from models.mixlinear.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    period_len: int = 24
    com_len: int = 4
    lpf: int = 1
    alpha: float = 0.5


def build_model(cfg, params):
    """Construct MixLinear from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], period_len=params.get('period_len', 24), com_len=params.get('com_len', 4), lpf=params.get('lpf', 1), alpha=params.get('alpha', 0.5))
    )


SPEC = ModelSpec(
    name='MixLinear',
    module='models.mixlinear',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/MixLinear.toml',
    model_card='src/models/mixlinear/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
