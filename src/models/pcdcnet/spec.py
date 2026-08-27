"""Model specification for PCDCNet."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.pcdcnet.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated PCDCNet parameters supplied via ``model.params``."""

    enc_in: int
    d_model: int = 64
    dropout: float = 0.1
    cov_dim: int | None = None


def build_model(cfg, params):
    """Construct PCDCNet from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], adj_mx=params.get('adj_mx'), cov_dim=params.get('cov_dim'), d_model=params.get('d_model', 64), dropout=params.get('dropout', 0.1))
    )


SPEC = ModelSpec(
    name='PCDCNet',
    module='models.pcdcnet',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='PCDCNet: A Surrogate Model for Air Quality Forecasting with Physical-Chemical Dynamics and Constraints',
        venue='arXiv preprint',
        year=2025,
        url='https://arxiv.org/abs/2505.19842',
    ),
    source=SourceRef(
        url='https://github.com/PoorOtterBob/CauAir',
        revision='73dae00ca6ad14abb15174a0a0286d500e868b94',
        license='NOASSERTION',
    ),
    evidence="unverified",
    config_path='configs/models/PCDCNet.toml',
    model_card='src/models/pcdcnet/README.md',
    smoke_config=None,
    capabilities=frozenset(['covariate']),
    components=('marks',),
    deviations=(
        'No author-released PCDCNet implementation was identified; the immediate CauAir source declares no code license.',
        'The local network contains residual feature mixing, graph propagation, and recurrent accumulation, but does not implement the paper emission inventory or chemical-constraint pipeline.',
        'Repository adjacency and shared calendar marks replace the paper station graph, meteorology, emissions, and numerical-model inputs.',
        'The common runner objective does not reproduce the paper physical-chemical constraint losses or 72-hour protocol.',
    ),
    contract_task={'seq_len': 24, 'pred_len': 24, 'label_len': 0},
)
