"""Model specification for CMoS."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
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
    paper=PaperRef(
        title='CMoS: Rethinking Time Series Prediction Through the Lens of Chunk-wise Spatial Correlations',
        venue='ICML 2025',
        year=2025,
        url='https://arxiv.org/abs/2505.19090',
    ),
    source=SourceRef(url='https://github.com/CSTCloudOps/CMoS', revision='b696a0c33b5ad8f03ad483d43b95fcb5564aa939', license='NOASSERTION'),
    evidence="unverified",
    config_path='configs/models/CMoS.toml',
    model_card='src/models/cmos/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    deviations=(
        'Chunk-wise linear correlation mappings, channel-wise convolutional routing, and correlation mixing were compared with model/CMoS/Model.py in the pinned author repository.',
        'ModernTSF removes an unused extra mapping and inert dropout option, adds optional top-k routing, and validates segment divisibility and routing bounds.',
        'The author periodicity-injection initialization is not exposed by this adapter, so the paper configuration space is incomplete.',
        'The author repository has no explicit code license or checkpoint parity evidence; verification remains blocked.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
