"""Model specification for SVTime."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.svtime.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    period: int = 24
    patch_size: int = 6
    revin: bool = True
    affine: bool = False
    subtract_last: bool = False
    analysis_act: str = "relu"
    analysis_hidden: str = "512,256"


def build_model(cfg, params):
    """Construct SVTime from a validated run configuration."""
    return (
    Model(c_in=params['enc_in'], period=params.get('period', 24), seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, patch_size=params.get('patch_size', 6), revin=bool(params.get('revin', True)), affine=bool(params.get('affine', False)), subtract_last=bool(params.get('subtract_last', False)), analysis_act=params.get('analysis_act', 'relu'), analysis_hidden=params.get('analysis_hidden', '512,256'))
    )


SPEC = ModelSpec(
    name='SVTime',
    module='models.svtime',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='SVTime: Small Time Series Forecasting Models Informed by \\"Physics\\" of Large Vision Model Forecasters',
        venue='arXiv preprint',
        year=2025,
        url='https://arxiv.org/abs/2510.09780',
    ),
    source=SourceRef(),
    evidence="unverified",
    config_path='configs/models/SVTime.toml',
    model_card='src/models/svtime/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('revin',),
    deviations=(
        'No authoritative source repository or licensed reference implementation is linked by the paper or author project page as of the audit.',
        'The local implementation has not been numerically compared with published checkpoints or paper tables.',
        'Paper-specific training, constraints, preprocessing, and dataset defaults have not been independently verified against this implementation.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
