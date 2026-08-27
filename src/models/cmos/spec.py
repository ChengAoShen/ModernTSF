"""Model specification for CMoS."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec
from models.cmos.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    seg_size: int = 4
    num_map: int = 3
    kernel_size: int = 3
    conv_stride: int = 1
    topk: int = 3


def build_model(cfg, params):
    """Construct CMoS from a validated run configuration."""
    return (
    Model(c_in=params['enc_in'], seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, seg_size=params.get('seg_size', 4), num_map=params.get('num_map', 3), kernel_size=params.get('kernel_size', 3), conv_stride=params.get('conv_stride', 1), topk=params.get('topk', 3))
    )


SPEC = ModelSpec(
    name='CMoS',
    module='models.cmos',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/CMoS.toml',
    model_card='src/models/cmos/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
