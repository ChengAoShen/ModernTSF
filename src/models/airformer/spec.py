"""Model specification for AirFormer."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.airformer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated AirFormer parameters supplied via ``model.params``."""

    enc_in: int
    d_model: int = 32
    nhead: int = 2
    num_encoder_layers: int = 4
    dropout: float = 0.3
    cov_dim: int = 2


def build_model(cfg, params):
    """Construct AirFormer from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], cov_dim=params.get('cov_dim', 2), d_model=params.get('d_model', 32), nhead=params.get('nhead', 2), num_encoder_layers=params.get('num_encoder_layers', 4), dropout=params.get('dropout', 0.3))
    )


SPEC = ModelSpec(
    name='AirFormer',
    module='models.airformer',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='AirFormer: Predicting Nationwide Air Quality in China with Transformers',
        venue='AAAI 2023',
        year=2023,
        url='https://doi.org/10.1609/aaai.v37i12.26676',
    ),
    source=SourceRef(
        url='https://github.com/yoshall/airformer',
        revision='ef7d3933768490e3a06921b8eb0f837c61741194',
        license='',
    ),
    evidence="unverified",
    config_path='configs/models/AirFormer.toml',
    model_card='src/models/airformer/README.md',
    smoke_config=None,
    capabilities=frozenset(['covariate']),
    components=('marks',),
    deviations=(
        'The in-tree implementation was consolidated from the CauAir baseline rather than copied directly from the pinned official repository.',
        'Spatial DS-MSA and its dataset-specific dartboard partitions are disabled; generic 1x1 residual projections are used instead.',
        'The adapter returns point forecasts only and does not expose the official reconstruction output or KL-divergence training term.',
        'Generic time marks replace the official air-quality feature and preprocessing pipeline.',
        'The official repository has no declared license file at the pinned revision.',
    ),
    contract_task={'seq_len': 24, 'pred_len': 24, 'label_len': 0},
)
