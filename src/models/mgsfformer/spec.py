"""Model specification for MGSFformer."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.mgsfformer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated MGSFformer parameters supplied via ``model.params``."""

    enc_in: int
    IE_dim: int = 32
    dropout: float = 0.3
    num_head: int = 2


def build_model(cfg, params):
    """Construct MGSFformer from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], IE_dim=params.get('IE_dim', 32), dropout=params.get('dropout', 0.3), num_head=params.get('num_head', 2))
    )


SPEC = ModelSpec(
    name='MGSFformer',
    module='models.mgsfformer',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='MGSFformer: A Multi-Granularity Spatiotemporal Fusion Transformer for air quality prediction',
        venue='Information Fusion 2025',
        year=2025,
        url='https://doi.org/10.1016/j.inffus.2024.102607',
    ),
    source=SourceRef(
        url='https://github.com/GestaltCogTeam/MGSFformer',
        revision='ff665a422a0ae001cfdd1b60ec9b4338a5ab406e',
        license='NOASSERTION',
    ),
    evidence="unverified",
    config_path='configs/models/MGSFformer.toml',
    model_card='src/models/mgsfformer/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=(),
    deviations=(
        'The bundled core consolidates the official MGSFformer architecture, IE, STA, DF, and RevIN modules into one file and replaces the framework base class with explicit dimensions.',
        'The adapter consumes only historical target values; it does not consume historical or future exogenous covariates.',
        'The architecture requires the input length to be divisible by 24 for its five fixed temporal granularities.',
        'The pinned author repository contains no license file or other explicit code-license grant.',
        'Official data preprocessing, training objective, initialization pipeline, and numerical results are not reproduced.',
    ),
    contract_task={'seq_len': 24, 'pred_len': 24, 'label_len': 0},
)
