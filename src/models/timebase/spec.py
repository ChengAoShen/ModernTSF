"""Model specification for TimeBase."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.timebase.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    period_len: int = 24
    basis_num: int = 6
    individual: bool = False
    use_orthogonal: bool = True
    use_period_norm: bool = True


def build_model(cfg, params):
    """Construct TimeBase from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], period_len=params.get('period_len', 24), basis_num=params.get('basis_num', 6), individual=bool(params.get('individual', False)), use_orthogonal=bool(params.get('use_orthogonal', True)), use_period_norm=bool(params.get('use_period_norm', True)))
    )


SPEC = ModelSpec(
    name='TimeBase',
    module='models.timebase',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='TimeBase: The Power of Minimalism in Efficient Long-term Time Series Forecasting',
        venue='ICML 2025',
        year=2025,
        url='',
    ),
    source=SourceRef(),
    evidence="unverified",
    config_path='configs/models/TimeBase.toml',
    model_card='src/models/timebase/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    deviations=(),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
