"""Model specification for MTSMixer."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.mtsmixer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 256
    d_ff: int = 64
    e_layers: int = 2
    fac_T: bool = False
    fac_C: bool = False
    sampling: int = 2
    norm: bool = True
    individual: bool = False
    rev: bool = True


def build_model(cfg, params):
    """Construct MTSMixer from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, features=cfg.task.features, enc_in=params['enc_in'], d_model=params.get('d_model', 256), d_ff=params.get('d_ff', 64), e_layers=params.get('e_layers', 2), fac_T=bool(params.get('fac_T', False)), fac_C=bool(params.get('fac_C', False)), sampling=params.get('sampling', 2), norm=bool(params.get('norm', True)), individual=bool(params.get('individual', False)), rev=bool(params.get('rev', True)))
    )


SPEC = ModelSpec(
    name='MTSMixer',
    module='models.mtsmixer',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='MTS-Mixers: Multivariate Time Series Forecasting via Factorized Temporal and Channel Mixing',
        venue='arXiv preprint',
        year=2023,
        url='https://arxiv.org/abs/2302.04501',
    ),
    source=SourceRef(),
    evidence="unverified",
    config_path='configs/models/MTSMixer.toml',
    model_card='src/models/mtsmixer/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=('revin',),
    deviations=(),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
