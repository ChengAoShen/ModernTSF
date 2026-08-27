"""Model specification for Pathformer."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.pathformer.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    layer_nums: int = 2
    k: int = 2
    num_experts: int = 4
    # Flat list of length layer_nums * num_experts, reshaped to per-layer
    # patch sizes. Each value must divide seq_len evenly.
    patch_size_list: list[int] = [16, 12, 8, 6, 16, 12, 8, 6]
    d_model: int = 16
    d_ff: int = 64
    residual_connection: int = 1
    revin: bool = True
    batch_norm: bool = False


def build_model(cfg, params):
    """Construct Pathformer from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, features=cfg.task.features, enc_in=params['enc_in'], layer_nums=params.get('layer_nums', 2), k=params.get('k', 2), num_experts=params.get('num_experts', 4), patch_size_list=params.get('patch_size_list', [16, 12, 8, 6, 16, 12, 8, 6]), d_model=params.get('d_model', 16), d_ff=params.get('d_ff', 64), residual_connection=params.get('residual_connection', 1), revin=bool(params.get('revin', True)), batch_norm=bool(params.get('batch_norm', False)))
    )


SPEC = ModelSpec(
    name='Pathformer',
    module='models.pathformer',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Pathformer: Multi-scale Transformers with Adaptive Pathways for Time Series Forecasting',
        venue='ICLR 2024',
        year=2024,
        url='https://arxiv.org/abs/2402.05956',
    ),
    source=SourceRef(url='https://github.com/decisionintelligence/pathformer', revision='ea85d82932215e171357da47b3bc82d502344758', license='NOASSERTION'),
    evidence="unverified",
    config_path='configs/models/Pathformer.toml',
    model_card='src/models/pathformer/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    deviations=(
        'Adaptive multi-scale routing, intra/inter-patch attention, sparse expert dispatch, decomposition/Fourier helpers, and RevIN were compared with the pinned author repository.',
        'ModernTSF removes hard-coded CUDA placement and adapts the config-object constructor to a flat patch-size list validated against layer/expert counts.',
        'Unused end-MLP parameters and normalization modules for a disabled batch_norm branch are not constructed.',
        'The upstream MoE balance loss is discarded by the point-forecast interface, so the training objective is not paper-equivalent.',
        'The author repository has no explicit code license and no numerical parity evidence; verification remains blocked.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
