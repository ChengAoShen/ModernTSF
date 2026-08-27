"""Model specification for DUET."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec
from models.duet.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    n_heads: int = 4
    e_layers: int = 2
    d_ff: int = 64
    dropout: float = 0.1
    fc_dropout: float = 0.1
    activation: str = "gelu"
    moving_avg: int = 25
    num_experts: int = 4
    k: int = 2
    hidden_size: int = 64
    noisy_gating: bool = True
    CI: bool = True


def build_model(cfg, params):
    """Construct DUET from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, features=cfg.task.features, enc_in=params['enc_in'], d_model=params.get('d_model', 64), n_heads=params.get('n_heads', 4), e_layers=params.get('e_layers', 2), d_ff=params.get('d_ff', 64), dropout=params.get('dropout', 0.1), fc_dropout=params.get('fc_dropout', 0.1), activation=params.get('activation', 'gelu'), moving_avg=params.get('moving_avg', 25), num_experts=params.get('num_experts', 4), k=params.get('k', 2), hidden_size=params.get('hidden_size', 64), noisy_gating=bool(params.get('noisy_gating', True)), CI=bool(params.get('CI', True)))
    )


SPEC = ModelSpec(
    name='DUET',
    module='models.duet',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/DUET.toml',
    model_card='src/models/duet/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('autoformer_encdec', 'revin'),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
