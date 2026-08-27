"""Model specification for TimeMixer."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec
from models.timemixer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    c_out: int
    freq: str = "h"
    embed: str = "timeF"
    e_layers: int = 2
    d_model: int = 512
    d_ff: int = 2048
    down_sampling_window: int = 1
    down_sampling_layers: int = 0
    down_sampling_method: str | None = None
    channel_independence: bool = False
    moving_avg: int = 25
    top_k: int = 5
    dropout: float = 0.0
    use_norm: bool = True
    decomp_method: str = "moving_avg"


def build_model(cfg, params):
    """Construct TimeMixer from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], c_out=params['c_out'], e_layers=params.get('e_layers', 2), d_model=params.get('d_model', 512), d_ff=params.get('d_ff', 2048), down_sampling_window=params.get('down_sampling_window', 1), down_sampling_layers=params.get('down_sampling_layers', 0), down_sampling_method=params.get('down_sampling_method'), channel_independence=bool(params.get('channel_independence', False)), moving_avg=params.get('moving_avg', 25), embed=params.get('embed', 'timeF'), top_k=params.get('top_k', 5), dropout=params.get('dropout', 0.0), freq=params.get('freq', 'h'), use_norm=bool(params.get('use_norm', True)), decomp_method=params.get('decomp_method', 'moving_avg'))
    )


SPEC = ModelSpec(
    name='TimeMixer',
    module='models.timemixer',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/TimeMixer.toml',
    model_card='src/models/timemixer/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('autoformer_encdec', 'embed', 'revin'),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
