"""Model specification for PAttn."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.pattn.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 128
    n_heads: int = 8
    d_ff: int = 256
    patch_len: int = 16
    stride: int = 8
    dropout: float = 0.1
    factor: int = 3
    activation: str = "gelu"


def build_model(cfg, params):
    """Construct PAttn from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, features=cfg.task.features, enc_in=params['enc_in'], d_model=params.get('d_model', 128), n_heads=params.get('n_heads', 8), d_ff=params.get('d_ff', 256), patch_len=params.get('patch_len', 16), stride=params.get('stride', 8), dropout=params.get('dropout', 0.1), factor=params.get('factor', 3), activation=params.get('activation', 'gelu'))
    )


SPEC = ModelSpec(
    name='PAttn',
    module='models.pattn',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Are Language Models Actually Useful for Time Series Forecasting?',
        venue='NeurIPS 2024',
        year=2024,
        url='https://arxiv.org/abs/2406.16964',
    ),
    source=SourceRef(),
    evidence="unverified",
    config_path='configs/models/PAttn.toml',
    model_card='src/models/pattn/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('self_attention_family',),
    deviations=(),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
