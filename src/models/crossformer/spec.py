"""Model specification for Crossformer."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.crossformer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    n_heads: int = 4
    e_layers: int = 2
    d_ff: int = 128
    seg_len: int = 12
    win_size: int = 2
    factor: int = 10
    dropout: float = 0.1


def build_model(cfg, params):
    """Construct Crossformer from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, features=cfg.task.features, enc_in=params['enc_in'], d_model=params.get('d_model', 64), n_heads=params.get('n_heads', 4), e_layers=params.get('e_layers', 2), d_ff=params.get('d_ff', 128), seg_len=params.get('seg_len', 12), win_size=params.get('win_size', 2), factor=params.get('factor', 10), dropout=params.get('dropout', 0.1))
    )


SPEC = ModelSpec(
    name='Crossformer',
    module='models.crossformer',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting',
        venue='ICLR 2023',
        year=2023,
        url='https://openreview.net/forum?id=vSVLM2j9eie',
    ),
    source=SourceRef(url='https://github.com/Thinklab-SJTU/Crossformer', revision='c10c8eadb153d1dd9798250967747ca3ebb81383', license='Apache-2.0'),
    evidence="adaptation",
    config_path='configs/models/Crossformer.toml',
    model_card='src/models/crossformer/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('embed', 'self_attention_family'),
    deviations=(
        'The forecast path was adapted from THUML Time-Series-Library revision 4e938a1767106324dd753b2a44832bf870a0252e and compared structurally with the pinned author repository.',
        'DSW patch embedding, two-stage cross-time/cross-dimension attention, hierarchical segment merging, and multi-scale decoder prediction are retained.',
        'The configs-object and non-forecast tasks are removed; shared attention and embedding components replace the source-local copies.',
        'The runnable preset uses seq_len=96, pred_len=96, seg_len=12 and smaller widths/layer counts than the paper repository examples.',
        'The generic runner owns preprocessing, objective, optimizer, and evaluation; no checkpoint or numerical parity is claimed.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
