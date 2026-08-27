"""Model specification for QuantileDLinear."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.quantile_dlinear.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    kernel_size: int = 25
    individual: bool = False
    quantile_levels: list[float] | None = None


def build_model(cfg, params):
    """Construct QuantileDLinear from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], features=cfg.task.features, kernel_size=params.get('kernel_size', 25), individual=bool(params.get('individual', False)), quantile_levels=params.get('quantile_levels'))
    )


SPEC = ModelSpec(
    name='QuantileDLinear',
    module='models.quantile_dlinear',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Are Transformers Effective for Time Series Forecasting? (DLinear backbone)',
        venue='AAAI 2023',
        year=2023,
        url='https://arxiv.org/abs/2205.13504',
    ),
    source=SourceRef(
        url='https://github.com/cure-lab/LTSF-Linear',
        revision='0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6',
        license='Apache-2.0',
    ),
    evidence="adaptation",
    config_path='configs/models/QuantileDLinear.toml',
    model_card='src/models/quantile_dlinear/README.md',
    smoke_config='configs/runs/smoke_quantile_dlinear.toml',
    capabilities=frozenset(['quantile-output', 'time-series']),
    components=('dlinear', 'quantile_head'),
    deviations=(
        'This is a ModernTSF composition of the verified DLinear backbone and the shared monotone QuantileHead; it is not a model proposed by the DLinear paper.',
        'It emits non-crossing quantiles and must be trained with pinball loss rather than the point-forecast objective used by the cited paper.',
        'No paper-level probabilistic benchmark parity is claimed.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
