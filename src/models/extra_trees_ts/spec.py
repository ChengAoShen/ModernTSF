"""Model specification for ExtraTreesTS."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec
from models.extra_trees_ts.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    dropout: float = 0.1
    num_layers: int = 1
    num_estimators: int = 16
    tree_depth: int = 3
    num_prototypes: int = 32
    kernel_gamma: float = 0.1
    l1_penalty: float = 0.0
    l2_penalty: float = 0.0
    use_revin: bool = True


def build_model(cfg, params):
    """Construct ExtraTreesTS from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], d_model=params.get('d_model', 64), dropout=params.get('dropout', 0.1), num_layers=params.get('num_layers', 1), num_estimators=params.get('num_estimators', 16), tree_depth=params.get('tree_depth', 3), num_prototypes=params.get('num_prototypes', 32), kernel_gamma=params.get('kernel_gamma', 0.1), l1_penalty=params.get('l1_penalty', 0.0), l2_penalty=params.get('l2_penalty', 0.0), use_revin=bool(params.get('use_revin', True)))
    )


SPEC = ModelSpec(
    name='ExtraTreesTS',
    module='models.extra_trees_ts',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/ExtraTreesTS.toml',
    model_card='src/models/extra_trees_ts/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    adapter='differentiable-ml-tsf',
    components=(),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
