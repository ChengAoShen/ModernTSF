"""Model specification for TimesNet."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec
from models.timesnet.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    c_out: int
    freq: str = "h"
    embed: str = "timeF"
    d_model: int = 512
    e_layers: int = 2
    d_ff: int = 2048
    dropout: float = 0.1
    top_k: int = 5
    num_kernels: int = 6


def build_model(cfg, params):
    """Construct TimesNet from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, label_len=cfg.task.label_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], c_out=params['c_out'], d_model=params.get('d_model', 512), e_layers=params.get('e_layers', 2), d_ff=params.get('d_ff', 2048), freq=params.get('freq', 'h'), dropout=params.get('dropout', 0.1), embed=params.get('embed', 'timeF'), top_k=params.get('top_k', 5), num_kernels=params.get('num_kernels', 6))
    )


SPEC = ModelSpec(
    name='TimesNet',
    module='models.timesnet',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/TimesNet.toml',
    model_card='src/models/timesnet/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('conv_blocks', 'dominant_periods', 'embed'),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
