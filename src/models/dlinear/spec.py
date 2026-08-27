"""Model specification for DLinear."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.dlinear.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    kernel_size: int = 25
    individual: bool = False


def build_model(cfg, params):
    """Construct DLinear from a validated run configuration."""
    return (
    Model(c_in=params['enc_in'], seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, kernel_size=params.get('kernel_size', 25), individual=params.get('individual', False))
    )


SPEC = ModelSpec(
    name='DLinear',
    module='models.dlinear',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Are Transformers Effective for Time Series Forecasting?',
        venue='AAAI 2023',
        year=2023,
        url='https://arxiv.org/abs/2205.13504',
    ),
    source=SourceRef(),
    evidence="unverified",
    config_path='configs/models/DLinear.toml',
    model_card='src/models/dlinear/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('dlinear',),
    deviations=(),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
