"""Model specification for MultiPatchFormer."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec
from models.multipatchformer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    n_heads: int = 4
    e_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.1


def build_model(cfg, params):
    """Construct MultiPatchFormer from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, label_len=cfg.task.label_len, features=cfg.task.features, enc_in=params['enc_in'], d_model=params.get('d_model', 64), n_heads=params.get('n_heads', 4), e_layers=params.get('e_layers', 2), d_ff=params.get('d_ff', 128), dropout=params.get('dropout', 0.1))
    )


SPEC = ModelSpec(
    name='MultiPatchFormer',
    module='models.multipatchformer',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/MultiPatchFormer.toml',
    model_card='src/models/multipatchformer/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('self_attention_family',),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
