"""Model specification for MultiPatchFormer."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.multipatchformer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    n_heads: int = 4
    e_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.1


def build_model(cfg, params):
    """Construct MultiPatchFormer from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, label_len=cfg.task.label_len, features=cfg.task.features, enc_in=params['enc_in'], d_model=params.get('d_model', 64), n_heads=params.get('n_heads', 4), e_layers=params.get('e_layers', 2), d_ff=params.get('d_ff', 128), dropout=params.get('dropout', 0.1))
    )


SPEC = ModelSpec(
    name='MultiPatchFormer',
    module='models.multipatchformer',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='A multiscale model for multivariate time series forecasting',
        venue='Scientific Reports 2025',
        year=2025,
        url='https://doi.org/10.1038/s41598-024-82417-4',
    ),
    source=SourceRef(url='https://github.com/thuml/Time-Series-Library', revision='4e938a1767106324dd753b2a44832bf870a0252e', license='MIT'),
    evidence="adaptation",
    config_path='configs/models/MultiPatchFormer.toml',
    model_card='src/models/multipatchformer/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('self_attention_family',),
    deviations=('Four parallel patch embeddings, temporal attention, channel-wise encoding, and the eight-stage semi-autoregressive head are retained.', 'The licensed Time-Series-Library port is used as the implementation source; bioinfoUQAM/MultiPatchFormer@965e6bd60822d509183253ef9c51fc3f9efe23f3 has no license file.', 'Only the single channel encoder executed by upstream is registered, and its unused remap layer is omitted.'),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
