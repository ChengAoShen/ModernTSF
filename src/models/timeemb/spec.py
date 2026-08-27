"""Model specification for TimeEmb."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.timeemb.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 512
    use_revin: bool = True
    use_hour_index: bool = True
    use_day_index: bool = False
    scale: float = 0.02
    hour_length: int = 24
    day_length: int = 7
    rec_lambda: float = 0.0
    auxi_lambda: float = 1.0
    auxi_loss: str = "MAE"
    auxi_mode: str = "fft"
    auxi_type: str = "complex"
    module_first: bool = True
    leg_degree: int = 2
    add_noise: bool = False


def build_model(cfg, params):
    """Construct TimeEmb from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], d_model=params.get('d_model', 512), use_revin=bool(params.get('use_revin', True)), use_hour_index=bool(params.get('use_hour_index', True)), use_day_index=bool(params.get('use_day_index', False)), scale=params.get('scale', 0.02), hour_length=params.get('hour_length', 24), day_length=params.get('day_length', 7))
    )


SPEC = ModelSpec(
    name='TimeEmb',
    module='models.timeemb',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='TimeEmb: A Lightweight Static-Dynamic Disentanglement Framework for Time Series Forecasting',
        venue='arXiv preprint',
        year=2025,
        url='https://arxiv.org/abs/2510.00461',
    ),
    source=SourceRef(),
    evidence="unverified",
    config_path='configs/models/TimeEmb.toml',
    model_card='src/models/timeemb/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    deviations=(),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
