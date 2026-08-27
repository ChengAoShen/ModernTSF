"""Model specification for TSMixer."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec
from models.tsmixer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    e_layers: int = 2
    dropout: float = 0.1


def build_model(cfg, params):
    """Construct TSMixer from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], d_model=params.get('d_model', 64), e_layers=params.get('e_layers', 2), dropout=params.get('dropout', 0.1))
    )


SPEC = ModelSpec(
    name='TSMixer',
    module='models.tsmixer',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/TSMixer.toml',
    model_card='src/models/tsmixer/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
