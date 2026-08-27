"""Model specification for DeepAR."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.deepar.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    embedding_size: int = 32
    hidden_size: int = 64
    num_layers: int = 2
    cov_feat_size: int = 0
    dropout: float = 0.1


def build_model(cfg, params):
    """Construct DeepAR from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, label_len=cfg.task.label_len, features=cfg.task.features, enc_in=params['enc_in'], embedding_size=params.get('embedding_size', 32), hidden_size=params.get('hidden_size', 64), num_layers=params.get('num_layers', 2), cov_feat_size=params.get('cov_feat_size', 0), dropout=params.get('dropout', 0.1))
    )


SPEC = ModelSpec(
    name='DeepAR',
    module='models.deepar',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks',
        venue='International Journal of Forecasting 2020',
        year=2020,
        url='https://arxiv.org/abs/1704.04110',
    ),
    source=SourceRef(),
    evidence="unverified",
    config_path='configs/models/DeepAR.toml',
    model_card='src/models/deepar/README.md',
    smoke_config=None,
    capabilities=frozenset(['distribution-output', 'time-series']),
    components=(),
    deviations=(),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
