"""Model specification for AirCade."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec
from models.aircade.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated AirCade parameters supplied via ``model.params``."""

    enc_in: int
    cov_dim: int | None = None
    input_embedding_dim: int = 16
    adaptive_embedding_dim: int = 24
    feed_forward_dim: int = 64
    num_heads: int = 4
    num_layers: int = 1
    node_embed_dim: int = 10


def build_model(cfg, params):
    """Construct AirCade from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], cov_dim=params.get('cov_dim'), input_embedding_dim=params.get('input_embedding_dim', 16), adaptive_embedding_dim=params.get('adaptive_embedding_dim', 24), feed_forward_dim=params.get('feed_forward_dim', 64), num_heads=params.get('num_heads', 4), num_layers=params.get('num_layers', 1), node_embed_dim=params.get('node_embed_dim', 10))
    )


SPEC = ModelSpec(
    name='AirCade',
    module='models.aircade',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/AirCade.toml',
    model_card='src/models/aircade/README.md',
    smoke_config=None,
    capabilities=frozenset(['covariate']),
    components=('base', 'marks'),
    contract_task={'seq_len': 24, 'pred_len': 24, 'label_len': 0},
)
