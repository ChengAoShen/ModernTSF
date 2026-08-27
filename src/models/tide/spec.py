"""Model specification for TiDE."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.tide.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    d_model: int = 512
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 2048
    decoder_output_dim: int = 7
    time_feat_dim: int = 6
    dropout: float = 0.1
    bias: bool = True
    feature_encode_dim: int = 2


def build_model(cfg, params):
    """Construct TiDE from a validated run configuration."""
    return Model(
        seq_len=cfg.task.seq_len,
        pred_len=cfg.task.pred_len,
        d_model=params.get('d_model', 512),
        e_layers=params.get('e_layers', 2),
        d_layers=params.get('d_layers', 1),
        d_ff=params.get('d_ff', 2048),
        decoder_output_dim=params.get('decoder_output_dim', 7),
        time_feat_dim=params.get('time_feat_dim', 6),
        dropout=params.get('dropout', 0.1),
        bias=bool(params.get('bias', True)),
        feature_encode_dim=params.get('feature_encode_dim', 2),
    )


SPEC = ModelSpec(
    name='TiDE',
    module='models.tide',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Long-term Forecasting with TiDE: Time-series Dense Encoder',
        venue='TMLR 2023',
        year=2023,
        url='https://arxiv.org/abs/2304.08424',
    ),
    source=SourceRef(
        url='https://github.com/thuml/Time-Series-Library',
        revision='4e938a1767106324dd753b2a44832bf870a0252e',
        license='MIT',
    ),
    evidence="adaptation",
    config_path='configs/models/TiDE.toml',
    model_card='src/models/tide/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    deviations=(
        'The local structure is adapted from the pinned THUML reference, not the official Google Research JAX implementation.',
        'The scalar temporal decoder omits LayerNorm: normalizing a one-dimensional output in the reference makes the nonlinear branch identically zero.',
        'Channels are forecast independently with shared weights; decoder_output_dim is an internal decoder width and time_feat_dim follows the runner marker contract.',
        'Classification, anomaly detection, imputation, Google preprocessing, and paper benchmark parity are outside this adapter.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
