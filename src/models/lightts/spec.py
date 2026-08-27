"""Model specification for LightTS."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.lightts.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    hid_dim: int = 128
    dropout: float = 0.0
    chunk_size: int = 24


def build_model(cfg, params):
    """Construct LightTS from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], hid_dim=params.get('hid_dim', 128), dropout=params.get('dropout', 0.0), chunk_size=params.get('chunk_size', 24))
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
    source=SourceRef(url='https://github.com/d-gcc/LightTS', revision='362ca172791559766f6a055be8f2cbed1bad5530', license='NOASSERTION'),
    evidence="unverified",
    config_path='configs/models/LightTS.toml',
    model_card='src/models/lightts/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    deviations=(
        'The local PyTorch baseline implements interval sampling, continuous sampling, information-exchange blocks, chunk aggregation, and an autoregressive highway structurally associated with LightTS.',
        'A traceable exact file-level derivation from the pinned author repository or THUML Time-Series-Library has not been established.',
        'seq_len must be divisible by chunk_size; the runnable preset uses 96 and 24 and validation prevents the upstream-style silent truncation of the lookback.',
        'The pinned author repository contains no explicit code license, and no checkpoint or numerical parity evidence is available.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
