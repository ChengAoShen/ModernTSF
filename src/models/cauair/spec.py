"""Model specification for CauAir."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec
from models.cauair.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated CauAir parameters supplied via ``model.params``."""

    enc_in: int
    cov_dim: int | None = None
    dim: int = 64
    rank: int = 8
    head: int = 4


def build_model(cfg, params):
    """Construct CauAir from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], cov_dim=params.get('cov_dim'), dim=params.get('dim', 64), rank=params.get('rank', 8), head=params.get('head', 4))
    )


SPEC = ModelSpec(
    name='CauAir',
    module='models.cauair',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/CauAir.toml',
    model_card='src/models/cauair/README.md',
    smoke_config=None,
    capabilities=frozenset(['covariate']),
    components=('base', 'marks'),
    contract_task={'seq_len': 24, 'pred_len': 24, 'label_len': 0},
)
