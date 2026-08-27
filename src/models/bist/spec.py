"""Model specification for BiST."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec
from models.bist.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated BiST parameters supplied via ``model.params``."""

    enc_in: int
    model_dim: int = 32
    prompt_dim: int = 32
    num_layer: int = 2
    hid_dim: int = 64
    tod_size: int = 24
    kernel_size: int = 3
    rp_layer: int = 1
    adaptive_adj_dim: int = 10
    core: int = 0


def build_model(cfg, params):
    """Construct BiST from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], model_dim=params.get('model_dim', 32), prompt_dim=params.get('prompt_dim', 32), num_layer=params.get('num_layer', 2), hid_dim=params.get('hid_dim', 64), tod_size=params.get('tod_size', 24), kernel_size=params.get('kernel_size', 3), rp_layer=params.get('rp_layer', 1), adaptive_adj_dim=params.get('adaptive_adj_dim', 10), core=params.get('core', 0))
    )


SPEC = ModelSpec(
    name='BiST',
    module='models.bist',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/BiST.toml',
    model_card='src/models/bist/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('base', 'marks', 'series_decomposition'),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)
