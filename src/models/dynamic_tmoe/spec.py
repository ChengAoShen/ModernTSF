"""Model specification for DynamicTMoE."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.dynamic_tmoe.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    dropout: float = 0.1
    period: int = 24
    num_prompts: int = 4
    use_revin: bool = True


def build_model(cfg, params):
    """Construct DynamicTMoE from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], d_model=params.get('d_model', 64), dropout=params.get('dropout', 0.1), period=params.get('period', 24), num_prompts=params.get('num_prompts', 4), use_revin=bool(params.get('use_revin', True)))
    )


SPEC = ModelSpec(
    name='DynamicTMoE',
    module='models.dynamic_tmoe',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Dynamic TMoE: A Drift-Aware Dynamic Mixture of Experts Framework for Non-Stationary Time Series Forecasting',
        venue='ICML 2026',
        year=2026,
        url='',
    ),
    source=SourceRef(),
    evidence="adaptation",
    config_path='configs/models/DynamicTMoE.toml',
    model_card='src/models/dynamic_tmoe/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    adapter='recent-tsf',
    components=(),
    deviations=('Uses the shared RecentTSFModel adapter and is not a paper reproduction.',),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
