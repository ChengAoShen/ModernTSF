"""Model specification for DSFormer."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.dsformer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    num_layer: int = 1
    muti_head: int = 2
    num_samp: int = 2
    dropout: float = 0.15
    if_node: bool = True


def build_model(cfg, params):
    """Construct DSFormer from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, label_len=cfg.task.label_len, features=cfg.task.features, enc_in=params['enc_in'], num_layer=params.get('num_layer', 1), muti_head=params.get('muti_head', 2), num_samp=params.get('num_samp', 2), dropout=params.get('dropout', 0.15), if_node=bool(params.get('if_node', True)))
    )


SPEC = ModelSpec(
    name='DSFormer',
    module='models.dsformer',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='DSformer: A Double Sampling Transformer for Multivariate Time Series Long-term Prediction',
        venue='CIKM 2023',
        year=2023,
        url='https://arxiv.org/abs/2308.03274',
    ),
    source=SourceRef(url='https://github.com/ChengqingYu/DSformer', revision='ccdbc354603e7842a89603649b0e33a8142c7701', license='NOASSERTION'),
    evidence="unverified",
    config_path='configs/models/DSFormer.toml',
    model_card='src/models/dsformer/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('revin',),
    deviations=(
        'Double down/piecewise sampling, temporal-variable attention, parallel feature paths, and the generative decoder were compared with the pinned author release.',
        'ModernTSF replaces the BasicTS-style constructor and reuses the shared RevIN implementation while retaining the forecast tensor layout.',
        'The author repository has no explicit code license and publishes no parity checkpoint for this adapter; evidence remains unverified.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
