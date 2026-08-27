"""Model specification for GCLSTM."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec
from models.gclstm.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated GCLSTM parameters supplied via ``model.params``."""

    enc_in: int
    cov_dim: int = 2
    Ks: int = 2


def build_model(cfg, params):
    """Construct GCLSTM from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], adj_mx=params.get('adj_mx'), cov_dim=params.get('cov_dim', 2), Ks=params.get('Ks', 2))
    )


SPEC = ModelSpec(
    name='GCLSTM',
    module='models.gclstm',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/GCLSTM.toml',
    model_card='src/models/gclstm/README.md',
    smoke_config=None,
    capabilities=frozenset(['covariate']),
    components=('graph_utils', 'marks'),
    contract_task={'seq_len': 24, 'pred_len': 24, 'label_len': 0},
)
