"""Model specification for MixLinear."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
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
    paper=PaperRef(
        title='MixLinear: Extreme Low Resource Multivariate Time Series Forecasting with 0.1K Parameters',
        venue='ICLR 2026',
        year=2026,
        url='https://arxiv.org/abs/2410.02081',
    ),
    source=SourceRef(
        url='https://github.com/aitianma/MixLinear',
        revision='42dbb98a5bbe64c13bc75b3cc07a9dc4acf20106',
        license='NOASSERTION',
    ),
    evidence="unverified",
    config_path='configs/models/MixLinear.toml',
    model_card='src/models/mixlinear/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    deviations=(
        'The pinned author repository contains no license file or other explicit code-license grant.',
        'The local time-domain branch uses two sequential low-rank projections controlled by com_len rather than the upstream pair of square-factorized intra/inter-segment transforms.',
        'The local convolution kernel and segment handling differ from the pinned source, so architecture equivalence is not established.',
        'A previously registered but unused full segment projection was removed from the local model.',
        'No checkpoint-level or numerical parity result is recorded.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
