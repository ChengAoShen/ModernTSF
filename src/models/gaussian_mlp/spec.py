"""Model specification for GaussianMLP."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.gaussian_mlp.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    hidden_size: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    eps: float = 1e-6


def build_model(cfg, params):
    """Construct GaussianMLP from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], features=cfg.task.features, hidden_size=params.get('hidden_size', 256), num_layers=params.get('num_layers', 2), dropout=params.get('dropout', 0.1), eps=params.get('eps', 1e-06))
    )


SPEC = ModelSpec(
    name='GaussianMLP',
    module='models.gaussian_mlp',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Gaussian-head MLP (ModernTSF parametric probabilistic baseline)',
        venue='ModernTSF',
        year=2026,
        url='',
    ),
    source=SourceRef(),
    evidence="adaptation",
    config_path='configs/models/GaussianMLP.toml',
    model_card='src/models/gaussian_mlp/README.md',
    smoke_config='configs/runs/smoke_gaussian_mlp.toml',
    capabilities=frozenset(['distribution-output', 'time-series']),
    components=(),
    deviations=(
        'This is an intentional in-repository probabilistic baseline, not a reproduction of an external paper or upstream repository.',
        'The MLP predicts Gaussian location and positive scale parameters and must be trained with Gaussian negative log likelihood.',
        'The independent Gaussian likelihood does not model cross-channel or cross-horizon covariance.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
