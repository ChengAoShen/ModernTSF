"""Model specification for HDMixer."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec
from models.hdmixer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 128
    d_ff: int = 256
    e_layers: int = 3
    patch_len: int = 16
    stride: int = 8
    dropout: float = 0.1
    head_dropout: float = 0.0
    activation: str = "gelu"
    individual: bool = False
    revin: bool = True
    affine: bool = True
    subtract_last: bool = False
    deform_range: float = 0.25
    mix_time: bool = True
    mix_variable: bool = True
    mix_channel: bool = True


def build_model(cfg, params):
    """Construct HDMixer from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, features=cfg.task.features, enc_in=params['enc_in'], d_model=params.get('d_model', 128), d_ff=params.get('d_ff', 256), e_layers=params.get('e_layers', 3), patch_len=params.get('patch_len', 16), stride=params.get('stride', 8), dropout=params.get('dropout', 0.1), head_dropout=params.get('head_dropout', 0.0), activation=params.get('activation', 'gelu'), individual=bool(params.get('individual', False)), revin=bool(params.get('revin', True)), affine=bool(params.get('affine', True)), subtract_last=bool(params.get('subtract_last', False)), deform_range=params.get('deform_range', 0.25), mix_time=bool(params.get('mix_time', True)), mix_variable=bool(params.get('mix_variable', True)), mix_channel=bool(params.get('mix_channel', True)))
    )


SPEC = ModelSpec(
    name='HDMixer',
    module='models.hdmixer',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/HDMixer.toml',
    model_card='src/models/hdmixer/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('flatten_forecast_head', 'revin'),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
