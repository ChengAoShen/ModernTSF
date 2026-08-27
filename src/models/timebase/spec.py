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
    orthogonal_weight: float = 0.08
    use_period_norm: bool = True


def build_model(cfg, params):
    """Construct TimeBase from a validated run configuration."""
    return (
        Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], period_len=params.get('period_len', 24), basis_num=params.get('basis_num', 6), individual=bool(params.get('individual', False)), orthogonal_weight=float(params.get('orthogonal_weight', 0.08)), use_period_norm=bool(params.get('use_period_norm', True)))
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
        url='https://proceedings.mlr.press/v267/huang25az.html',
    ),
    source=SourceRef(),
    evidence="paper-reimplementation",
    config_path='configs/models/TimeBase.toml',
    model_card='src/models/timebase/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    deviations=(
        'No author code repository or redistributable upstream implementation was established; verification is against the ICML paper equations only.',
        'The model implements segment padding, basis extraction, segment forecasting, period normalization, and Eq. 6 orthogonality regularization.',
        'orthogonal_weight is wired to trainer aux_loss (Eq. 7); 0.08 is a runnable preset within the paper sweep and must be tuned per dataset.',
        'Published dataset-specific settings and numerical parity remain unverified.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
