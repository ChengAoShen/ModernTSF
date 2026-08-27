"""Model specification for CrossLinear."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.crosslinear.model import Model

from typing import Optional

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    dec_in: Optional[int] = None
    patch_len: int = 16
    d_model: int = 32
    d_ff: int = 2048
    alpha: float = 1.0
    beta: float = 0.5


def build_model(cfg, params):
    """Construct CrossLinear from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, dec_in=params.get('dec_in') or params['enc_in'], patch_len=params.get('patch_len', 16), d_model=params.get('d_model', 32), d_ff=params.get('d_ff', 2048), alpha=params.get('alpha', 1.0), beta=params.get('beta', 0.5))
    )


SPEC = ModelSpec(
    name='CrossLinear',
    module='models.crosslinear',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='CrossLinear: Plug-and-Play Cross-Correlation Embedding for Time Series Forecasting with Exogenous Variables',
        venue='KDD 2025',
        year=2025,
        url='https://arxiv.org/abs/2505.23116',
    ),
    source=SourceRef(),
    evidence="unverified",
    config_path='configs/models/CrossLinear.toml',
    model_card='src/models/crosslinear/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    deviations=(),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
