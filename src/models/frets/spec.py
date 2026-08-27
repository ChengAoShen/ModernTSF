"""Model specification for FreTS."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.frets.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    embed_size: int = 128
    hidden_size: int = 256
    channel_independence: bool = False


def build_model(cfg, params):
    """Construct FreTS from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, features=cfg.task.features, enc_in=params['enc_in'], embed_size=params.get('embed_size', 128), hidden_size=params.get('hidden_size', 256), channel_independence=bool(params.get('channel_independence', False)))
    )


SPEC = ModelSpec(
    name='FreTS',
    module='models.frets',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Frequency-domain MLPs are More Effective Learners in Time Series Forecasting',
        venue='NeurIPS 2023',
        year=2023,
        url='https://arxiv.org/abs/2311.06184',
    ),
    source=SourceRef(),
    evidence="unverified",
    config_path='configs/models/FreTS.toml',
    model_card='src/models/frets/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    deviations=(),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
