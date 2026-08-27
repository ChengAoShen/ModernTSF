"""Model specification for PatchMLP."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.patchmlp.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 1024
    e_layers: int = 1
    use_norm: bool = True
    moving_avg: int = 13
    patch_len: list[int] = [48, 24, 12, 6]


def build_model(cfg, params):
    """Construct PatchMLP from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], d_model=params.get('d_model', 1024), e_layers=params.get('e_layers', 1), use_norm=bool(params.get('use_norm', True)), moving_avg=params.get('moving_avg', 13), patch_len=params.get('patch_len'))
    )


SPEC = ModelSpec(
    name='PatchMLP',
    module='models.patchmlp',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Unlocking the Power of Patch: Patch-Based MLP for Long-Term Time Series Forecasting',
        venue='AAAI 2025',
        year=2025,
        url='https://arxiv.org/abs/2405.13575',
    ),
    source=SourceRef(),
    evidence="paper-reimplementation",
    config_path='configs/models/PatchMLP.toml',
    model_card='src/models/patchmlp/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    deviations=(
        'No author source repository or pinned upstream implementation was established; the code is checked against the paper architecture only.',
        'The implementation covers multi-scale patch embedding, moving-average decomposition, temporal and channel MLP mixing, and the forecast projection.',
        'The second encoder LayerNorm is now applied to the final residual path; it was previously instantiated but unreachable.',
        'Published dataset-specific hyperparameters and numerical parity remain unverified.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
