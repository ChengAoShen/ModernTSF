"""Model specification for LightTS."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.lightts.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    hid_dim: int = 128
    dropout: float = 0.0
    chunk_size: int = 40
    c_dim: int = 40


def build_model(cfg, params):
    """Construct LightTS from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], hid_dim=params.get('hid_dim', 128), dropout=params.get('dropout', 0.0), chunk_size=params.get('chunk_size', 40), c_dim=params.get('c_dim', 40))
    )


SPEC = ModelSpec(
    name='LightTS',
    module='models.lightts',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Less Is More: Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP Structures',
        venue='arXiv preprint',
        year=2022,
        url='https://arxiv.org/abs/2207.01186',
    ),
    source=SourceRef(),
    evidence="unverified",
    config_path='configs/models/LightTS.toml',
    model_card='src/models/lightts/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    deviations=(),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
