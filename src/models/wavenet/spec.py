"""Model specification for WaveNet."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.wavenet.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    residual_channels: int = 16
    dilation_channels: int = 16
    skip_channels: int = 64
    end_channels: int = 128
    kernel_size: int = 2
    blocks: int = 2
    layers: int = 2
    use_norm: bool = True


def build_model(cfg, params):
    """Construct WaveNet from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, label_len=cfg.task.label_len, features=cfg.task.features, enc_in=params['enc_in'], residual_channels=params.get('residual_channels', 16), dilation_channels=params.get('dilation_channels', 16), skip_channels=params.get('skip_channels', 64), end_channels=params.get('end_channels', 128), kernel_size=params.get('kernel_size', 2), blocks=params.get('blocks', 2), layers=params.get('layers', 2), use_norm=bool(params.get('use_norm', True)))
    )


SPEC = ModelSpec(
    name='WaveNet',
    module='models.wavenet',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='WaveNet: A Generative Model for Raw Audio',
        venue='arXiv preprint',
        year=2016,
        url='https://arxiv.org/abs/1609.03499',
    ),
    source=SourceRef(),
    evidence="unverified",
    config_path='configs/models/WaveNet.toml',
    model_card='src/models/wavenet/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('revin',),
    deviations=(),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
