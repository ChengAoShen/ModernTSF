"""Model specification for Aurora."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.aurora.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    dropout: float = 0.1
    period: int = 24
    num_prompts: int = 4
    use_revin: bool = True


def build_model(cfg, params):
    """Construct Aurora from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], d_model=params.get('d_model', 64), dropout=params.get('dropout', 0.1), period=params.get('period', 24), num_prompts=params.get('num_prompts', 4), use_revin=bool(params.get('use_revin', True)))
    )


SPEC = ModelSpec(
    name='Aurora',
    module='models.aurora',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Aurora: Towards Universal Generative Multimodal Time Series Forecasting',
        venue='ICLR 2026',
        year=2026,
        url='https://arxiv.org/abs/2509.22295',
    ),
    source=SourceRef(),
    evidence="adaptation",
    config_path='configs/models/Aurora.toml',
    model_card='src/models/aurora/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    adapter='recent-tsf',
    components=(),
    deviations=('Uses the shared RecentTSFModel adapter and is not a paper reproduction.',),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
