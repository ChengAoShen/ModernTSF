"""Model specification for SparseTSF."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.sparsetsf.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    period: int = 24
    d_model: int = 64
    model_type: str = "linear"


def build_model(cfg, params):
    """Construct SparseTSF from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], period=params.get('period', 24), d_model=params.get('d_model', 64), model_type=params.get('model_type', 'linear'))
    )


SPEC = ModelSpec(
    name='SparseTSF',
    module='models.sparsetsf',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='SparseTSF: Modeling Long-term Time Series Forecasting with 1k Parameters',
        venue='ICML 2024',
        year=2024,
        url='https://arxiv.org/abs/2405.00946',
    ),
    source=SourceRef(
        url='https://github.com/lss-1138/SparseTSF',
        revision='b8c2740eecc84d8095ffce49ba5acafe68e53bb8',
        license='Apache-2.0',
    ),
    evidence="upstream-port",
    config_path='configs/models/SparseTSF.toml',
    model_card='src/models/sparsetsf/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    deviations=(
        'The public parameter name period maps to the upstream period_len setting.',
        'For non-divisible input lengths the adapter truncates to a complete trailing period; official experiment configs use divisible lengths.',
        'Official preprocessing, training schedules, and reported numerical results are not reproduced by the model package.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
