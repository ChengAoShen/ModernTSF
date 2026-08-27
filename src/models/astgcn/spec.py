"""Model specification for ASTGCN."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec
from models.astgcn.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated ASTGCN parameters supplied via ``model.params``."""

    enc_in: int
    cov_dim: int = 2
    nb_block: int = 2
    K: int = 3
    nb_chev_filter: int = 64
    nb_time_filter: int = 64


def build_model(cfg, params):
    """Construct ASTGCN from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], adj_mx=params.get('adj_mx'), cov_dim=params.get('cov_dim', 2), nb_block=params.get('nb_block', 2), K=params.get('K', 3), nb_chev_filter=params.get('nb_chev_filter', 64), nb_time_filter=params.get('nb_time_filter', 64))
    )


SPEC = ModelSpec(
    name='ASTGCN',
    module='models.astgcn',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/ASTGCN.toml',
    model_card='src/models/astgcn/README.md',
    smoke_config=None,
    capabilities=frozenset(['covariate']),
    components=('graph_utils', 'marks'),
    contract_task={'seq_len': 24, 'pred_len': 24, 'label_len': 0},
)
