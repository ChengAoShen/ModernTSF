"""Model specification for CrossLinear."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.crosslinear.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    patch_len: int = 16
    d_model: int = 32
    d_ff: int = 2048
    alpha: float = 1.0
    beta: float = 0.5


def build_model(cfg, params):
    """Construct CrossLinear from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], patch_len=params.get('patch_len', 16), d_model=params.get('d_model', 32), d_ff=params.get('d_ff', 2048), alpha=params.get('alpha', 1.0), beta=params.get('beta', 0.5))
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
    source=SourceRef(url='https://github.com/mumiao2000/CrossLinear', revision='d22366e2f59ced560a02b2b1c7cc673e3c02a13f', license='MIT'),
    evidence="adaptation",
    config_path='configs/models/CrossLinear.toml',
    model_card='src/models/crosslinear/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    deviations=(
        'Cross-correlation convolution, learnable direct/correlation and value/position blends, non-overlapping patch embedding, and global linear forecast head were compared with models/CrossLinear.py in the pinned MIT author repository.',
        'ModernTSF exposes the ordinary multivariate path only; the upstream MS target-channel mode and explicit exogenous-variable role selection are not available through the common interface.',
        'The duplicate dec_in setting was removed in favor of the catalog-standard enc_in channel count.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
