"""Model specification for LSTM."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec
from models.lstm.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    init_dim: int = 32
    hid_dim: int = 64
    end_dim: int = 128
    layer: int = 2
    dropout: float = 0.1
    cov_dim: int = 2


def build_model(cfg, params):
    """Construct LSTM from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], init_dim=params.get('init_dim', 32), hid_dim=params.get('hid_dim', 64), end_dim=params.get('end_dim', 128), layer=params.get('layer', 2), dropout=params.get('dropout', 0.1), cov_dim=params.get('cov_dim', 2))
    )


SPEC = ModelSpec(
    name='LSTM',
    module='models.lstm',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/LSTM.toml',
    model_card='src/models/lstm/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('marks',),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)
