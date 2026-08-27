"""Model specification for S4."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec
from models.s4.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 128
    d_state: int = 64
    e_layers: int = 2
    dropout: float = 0.1
    use_norm: bool = True


def build_model(cfg, params):
    """Construct S4 from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, label_len=cfg.task.label_len, features=cfg.task.features, enc_in=params['enc_in'], d_model=params.get('d_model', 128), d_state=params.get('d_state', 64), e_layers=params.get('e_layers', 2), dropout=params.get('dropout', 0.1), use_norm=bool(params.get('use_norm', True)))
    )


SPEC = ModelSpec(
    name='S4',
    module='models.s4',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/S4.toml',
    model_card='src/models/s4/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
