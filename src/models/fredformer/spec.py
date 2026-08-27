"""Model specification for Fredformer."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.fredformer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 16
    patch_len: int = 16
    stride: int = 8
    revin: bool = True
    affine: bool = True
    subtract_last: bool = False
    individual: bool = False
    head_dropout: float = 0.0
    cf_dim: int = 48
    cf_depth: int = 2
    cf_heads: int = 6
    cf_mlp: int = 128
    cf_head_dim: int = 32
    cf_drop: float = 0.2


def build_model(cfg, params):
    """Construct Fredformer from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, label_len=cfg.task.label_len, features=cfg.task.features, enc_in=params['enc_in'], d_model=params.get('d_model', 16), patch_len=params.get('patch_len', 16), stride=params.get('stride', 8), revin=bool(params.get('revin', True)), affine=bool(params.get('affine', True)), subtract_last=bool(params.get('subtract_last', False)), individual=bool(params.get('individual', False)), head_dropout=params.get('head_dropout', 0.0), cf_dim=params.get('cf_dim', 48), cf_depth=params.get('cf_depth', 2), cf_heads=params.get('cf_heads', 6), cf_mlp=params.get('cf_mlp', 128), cf_head_dim=params.get('cf_head_dim', 32), cf_drop=params.get('cf_drop', 0.2))
    )


SPEC = ModelSpec(
    name='Fredformer',
    module='models.fredformer',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Fredformer: Frequency Debiased Transformer for Time Series Forecasting',
        venue='KDD 2024',
        year=2024,
        url='https://arxiv.org/abs/2406.09009',
    ),
    source=SourceRef(url='https://github.com/chenzRG/Fredformer', revision='fa64775ea1012e313cbe30fe2c9b7e493a798aae', license='NOASSERTION'),
    evidence="unverified",
    config_path='configs/models/Fredformer.toml',
    model_card='src/models/fredformer/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('revin',),
    deviations=(
        'Frequency-domain real/imaginary patch construction, cross-frequency Transformer, RevIN, and forecast head were compared with the pinned author repository.',
        'The adapter keeps the standard non-Nystrom forecast path and removes generic e_layers, n_heads, d_ff, and dropout options that never reached that path, plus an unused LayerNorm.',
        'The author repository has no explicit code license and no checkpoint parity evidence; verification remains blocked.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
