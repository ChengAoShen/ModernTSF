"""Model specification for DTAF."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.dtaf.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 32
    e_layers: int = 1
    patch_len: int = 16
    stride: int = 8
    heads: int = 2
    dropout: float = 0.1
    moving_avg: int = 25
    expert_num: int = 2
    kan_div: int = 4
    k: int = 1
    aggregated_norm: int = 1


def build_model(cfg, params):
    """Construct DTAF from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, label_len=cfg.task.label_len, features=cfg.task.features, enc_in=params['enc_in'], d_model=params.get('d_model', 32), e_layers=params.get('e_layers', 1), patch_len=params.get('patch_len', 16), stride=params.get('stride', 8), heads=params.get('heads', 2), dropout=params.get('dropout', 0.1), moving_avg=params.get('moving_avg', 25), expert_num=params.get('expert_num', 2), kan_div=params.get('kan_div', 4), k=params.get('k', 1), aggregated_norm=params.get('aggregated_norm', 1))
    )


SPEC = ModelSpec(
    name='DTAF',
    module='models.dtaf',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Towards Non-Stationary Time Series Forecasting with Temporal Stabilization and Frequency Differencing',
        venue='AAAI 2026',
        year=2026,
        url='https://arxiv.org/abs/2511.08229',
    ),
    source=SourceRef(url='https://github.com/decisionintelligence/DTAF', revision='9d12aa4061c771b419c5a5bba9f2bf95d9419c41', license='NOASSERTION'),
    evidence="unverified",
    config_path='configs/models/DTAF.toml',
    model_card='src/models/dtaf/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('autoformer_encdec', 'embed'),
    deviations=(
        'Temporal stabilizing fusion with KAN mixture-of-experts, frequency differencing, dual attention, and fused prediction were compared with the pinned author repository.',
        'The local adapter removes upstream torch.save debug side effects and auxiliary stables output, and reuses shared patch embedding and decomposition layers.',
        'The pinned source calls frequency_attention for both temporal and frequency branches; ModernTSF routes H_t through temporal_attention as required by the named dual-branch architecture and removes the otherwise permanently dead parameters.',
        'The author repository has no explicit code license and no numerical parity evidence; verification remains blocked.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
