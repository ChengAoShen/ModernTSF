"""Model specification for Koopa."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.koopa.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    seg_len: int | None = None
    dynamic_dim: int = 128
    hidden_dim: int = 64
    hidden_layers: int = 2
    num_blocks: int = 3
    multistep: bool = False
    alpha: float = 0.2


def build_model(cfg, params):
    """Construct Koopa from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, label_len=cfg.task.label_len, features=cfg.task.features, enc_in=params['enc_in'], seg_len=params.get('seg_len', None), dynamic_dim=params.get('dynamic_dim', 128), hidden_dim=params.get('hidden_dim', 64), hidden_layers=params.get('hidden_layers', 2), num_blocks=params.get('num_blocks', 3), multistep=bool(params.get('multistep', False)), alpha=params.get('alpha', 0.2))
    )


SPEC = ModelSpec(
    name='Koopa',
    module='models.koopa',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors',
        venue='NeurIPS 2023',
        year=2023,
        url='https://arxiv.org/abs/2305.18803',
    ),
    source=SourceRef(),
    evidence="unverified",
    config_path='configs/models/Koopa.toml',
    model_card='src/models/koopa/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    deviations=(),
    # Koopa's default segment length equals pred_len and its DMD layer needs
    # at least two input segments.  This mirrors the upstream 2:1 setup.
    contract_task={'seq_len': 192, 'pred_len': 96, 'label_len': 0},
)
