"""Model specification for NLinear."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.nlinear.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    individual: bool = False


def build_model(cfg, params):
    """Construct NLinear from a validated run configuration."""
    return (
    Model(c_in=params['enc_in'], seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, individual=bool(params.get('individual', False)))
    )


SPEC = ModelSpec(
    name='NLinear',
    module='models.nlinear',
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
    config_path='configs/models/NLinear.toml',
    model_card='src/models/nlinear/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    deviations=(),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
