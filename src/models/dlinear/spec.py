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
    source=SourceRef(
        url='https://github.com/cure-lab/LTSF-Linear',
        revision='0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6',
        license='Apache-2.0',
    ),
    evidence="upstream-port",
    config_path='configs/models/DLinear.toml',
    model_card='src/models/dlinear/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('dlinear',),
    deviations=(
        'The official DLinear module is factored into the shared paper-neutral DLinear backbone and a named wrapper.',
        'The moving-average kernel is configurable locally; the preset and upstream implementation use 25.',
        'Paper-specific data preprocessing, training schedules, and reported numerical results are not reproduced by the model package.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
