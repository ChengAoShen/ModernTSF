"""Model specification for TimeFilter."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.timefilter.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    d_ff: int = 128
    n_heads: int = 4
    e_layers: int = 2
    patch_len: int = 16
    dropout: float = 0.1
    alpha: float = 0.1
    top_p: float = 0.5
    pos: bool = True


def build_model(cfg, params):
    """Construct TimeFilter from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, label_len=cfg.task.label_len, features=cfg.task.features, enc_in=params['enc_in'], d_model=params.get('d_model', 64), d_ff=params.get('d_ff', 128), n_heads=params.get('n_heads', 4), e_layers=params.get('e_layers', 2), patch_len=params.get('patch_len', 16), dropout=params.get('dropout', 0.1), alpha=params.get('alpha', 0.1), top_p=params.get('top_p', 0.5), pos=bool(params.get('pos', True)))
    )


SPEC = ModelSpec(
    name='TimeFilter',
    module='models.timefilter',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='TimeFilter: Patch-Specific Spatial-Temporal Graph Filtration for Time Series Forecasting',
        venue='ICML 2025',
        year=2025,
        url='https://arxiv.org/abs/2501.13041',
    ),
    source=SourceRef(url='https://github.com/TROUBADOUR000/TimeFilter', revision='dffde87e4fff0fdeeebbacde03dc1e432e15b3a1', license=''),
    evidence="unverified",
    config_path='configs/models/TimeFilter.toml',
    model_card='src/models/timefilter/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('embed', 'revin'),
    deviations=('Patch graph construction, spatial/temporal/joint region masks, graph filtration, and adaptive expert routing follow the author implementation.', 'The auxiliary MoE loss is exposed only as last_moe_loss because the common point-forecast trainer has no auxiliary-loss channel; its hard-routing weights are frozen.', 'The author repository has no license file at the pinned revision, so the model remains unverified.'),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
