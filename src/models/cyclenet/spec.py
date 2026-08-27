"""Model specification for CycleNet."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.cyclenet.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    cycle: int = 24
    model_type: str = "linear"
    d_model: int = 512
    use_revin: bool = True


def build_model(cfg, params):
    """Construct CycleNet from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], cycle=params.get('cycle', 24), model_type=params.get('model_type', 'linear'), d_model=params.get('d_model', 512), use_revin=bool(params.get('use_revin', True)))
    )


SPEC = ModelSpec(
    name='CycleNet',
    module='models.cyclenet',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='CycleNet: Enhancing Time Series Forecasting through Modeling Periodic Patterns',
        venue='NeurIPS 2024',
        year=2024,
        url='https://arxiv.org/abs/2409.18479',
    ),
    source=SourceRef(
        url='https://github.com/ACAT-SCUT/CycleNet',
        revision='d807e51fc2dcd143885ee639d97965a7ab0926f4',
        license='Apache-2.0',
    ),
    evidence="upstream-port",
    config_path='configs/models/CycleNet.toml',
    model_card='src/models/cyclenet/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    deviations=(
        'The adapter derives the first forecast-step cycle index from ModernTSF decoder calendar marks instead of receiving the upstream data-loader cycle_index tensor.',
        'Cycle lengths 24, 7, and 168 have explicit hour/weekday mappings; other cycles fall back to hour modulo cycle and may not represent the dataset phase.',
        'Official dataset-specific cycle selection, preprocessing, training, and numerical results are not reproduced by the model package.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
