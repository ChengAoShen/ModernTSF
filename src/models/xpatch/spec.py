"""Model specification for xPatch."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.xpatch.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    patch_len: int = 16
    stride: int = 8
    padding_patch: str = "end"
    ma_type: str = "ema"
    alpha: float = 0.3
    beta: float = 0.3
    revin: bool = True


def build_model(cfg, params):
    """Construct xPatch from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], patch_len=params.get('patch_len', 16), stride=params.get('stride', 8), padding_patch=params.get('padding_patch', 'end'), ma_type=params.get('ma_type', 'ema'), alpha=params.get('alpha', 0.3), beta=params.get('beta', 0.3), revin=bool(params.get('revin', True)))
    )


SPEC = ModelSpec(
    name='xPatch',
    module='models.xpatch',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='xPatch: Dual-Stream Time Series Forecasting with Exponential Seasonal-Trend Decomposition',
        venue='AAAI 2025',
        year=2025,
        url='https://arxiv.org/abs/2412.17323',
    ),
    source=SourceRef(url='https://github.com/stitsyuk/xPatch', revision='d12eecaa11409109582f5e2ffdebcc2cffd47b3e', license='Apache-2.0'),
    evidence="upstream-port",
    config_path='configs/models/xPatch.toml',
    model_card='src/models/xpatch/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('revin',),
    deviations=('EMA/DEMA seasonal-trend decomposition, nonlinear patch-CNN stream, linear trend stream, and final dual-stream fusion match the official model.', 'The configs-object constructor is replaced by explicit parameters and the shared RevIN implementation.', 'The repository model does not implement the paper custom arctangent loss or sigmoid learning-rate schedule; those training choices remain outside this model port.'),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
