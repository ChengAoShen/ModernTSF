"""Model specification for SOFTS."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.softs.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 128
    d_core: int = 64
    d_ff: int = 256
    e_layers: int = 2
    dropout: float = 0.1
    activation: str = "gelu"
    use_norm: bool = True


def build_model(cfg, params):
    """Construct SOFTS from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, label_len=cfg.task.label_len, features=cfg.task.features, enc_in=params['enc_in'], d_model=params.get('d_model', 128), d_core=params.get('d_core', 64), d_ff=params.get('d_ff', 256), e_layers=params.get('e_layers', 2), dropout=params.get('dropout', 0.1), activation=params.get('activation', 'gelu'), use_norm=bool(params.get('use_norm', True)))
    )


SPEC = ModelSpec(
    name='SOFTS',
    module='models.softs',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion',
        venue='NeurIPS 2024',
        year=2024,
        url='https://proceedings.neurips.cc/paper_files/paper/2024/hash/754612bde73a8b65ad8743f1f6d8ddf6-Abstract-Conference.html',
    ),
    source=SourceRef(url='https://github.com/Secilia-Cxy/SOFTS', revision='f5d35fd7c3e716b6383ce6d3cc42c131e32c3c44', license='MIT'),
    evidence="adaptation",
    config_path='configs/models/SOFTS.toml',
    model_card='src/models/softs/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('embed', 'transformer_encdec'),
    deviations=(
        'The forecast path was adapted from THUML Time-Series-Library revision 4e938a1767106324dd753b2a44832bf870a0252e and compared with the pinned author repository.',
        'Inverted variate embedding, stochastic/deterministic STAR core aggregation, redistribution, stacked encoder layers, projection, and normalization are retained.',
        'The local STAR module is fitted into the shared Transformer encoder interface and non-forecast tasks/upstream experiment code are omitted.',
        'No checkpoint or published numerical parity is claimed.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
