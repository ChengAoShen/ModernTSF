"""Model specification for Sumba."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.sumba.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    input_dim: int = 1
    output_dim: int = 1
    residual_channels: int = 16
    conv_channels: int = 16
    skip_channels: int = 32
    end_channels: int = 64
    dimension: int = 16
    M: int = 4
    LowRank: int = 8
    D: int = 16
    gcn_depth: int = 2
    sumba_layers: int = 2
    layers: int = 2
    dilation_exponential: int = 1
    kernel_set: list[int] = [2, 3, 6, 7]
    propalpha: float = 0.05
    dropout: float = 0.3
    layer_norm_affline: bool = True
    mark_dim: int = 6


def build_model(cfg, params):
    """Construct Sumba from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, label_len=cfg.task.label_len, features=cfg.task.features, enc_in=params['enc_in'], input_dim=params.get('input_dim', 1), output_dim=params.get('output_dim', 1), residual_channels=params.get('residual_channels', 16), conv_channels=params.get('conv_channels', 16), skip_channels=params.get('skip_channels', 32), end_channels=params.get('end_channels', 64), dimension=params.get('dimension', 16), M=params.get('M', 4), LowRank=params.get('LowRank', 8), D=params.get('D', 16), gcn_depth=params.get('gcn_depth', 2), sumba_layers=params.get('sumba_layers', 2), layers=params.get('layers', 2), dilation_exponential=params.get('dilation_exponential', 1), kernel_set=tuple(params.get('kernel_set', (2, 3, 6, 7))), propalpha=params.get('propalpha', 0.05), dropout=params.get('dropout', 0.3), layer_norm_affline=bool(params.get('layer_norm_affline', True)), mark_dim=params.get('mark_dim', 6))
    )


SPEC = ModelSpec(
    name='Sumba',
    module='models.sumba',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Structured Matrix Basis for Multivariate Time Series Forecasting with Interpretable Dynamics',
        venue='NeurIPS 2024',
        year=2024,
        url='',
    ),
    source=SourceRef(),
    evidence="unverified",
    config_path='configs/models/Sumba.toml',
    model_card='src/models/sumba/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    deviations=(),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
)
