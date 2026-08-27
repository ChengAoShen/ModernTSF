"""Model specification for STPGNN."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.stpgnn.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated STPGNN parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset and
    must not be declared in the TOML. ``enc_in`` (= number of nodes ``N``) is the
    required spatial dimension; the remaining fields are kept small so the smoke
    run stays fast.
    """

    enc_in: int
    input_dim: int = 3
    dropout: float = 0.1
    topk: int = 4
    residual_channels: int = 16
    dilation_channels: int = 16
    end_channels: int = 64
    kernel_size: int = 2
    blocks: int = 2
    layers: int = 2
    days: int = 7
    time_of_day_size: int = 24
    dims: int = 16
    order: int = 2
    normalization: str = "batch"


def build_model(cfg, params):
    """Construct STPGNN from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, num_nodes=params.get('num_nodes', params['enc_in']), adj_mx=params.get('adj_mx'), input_dim=params.get('input_dim', 3), dropout=params.get('dropout', 0.1), topk=params.get('topk', 4), residual_channels=params.get('residual_channels', 16), dilation_channels=params.get('dilation_channels', 16), end_channels=params.get('end_channels', 64), kernel_size=params.get('kernel_size', 2), blocks=params.get('blocks', 2), layers=params.get('layers', 2), days=params.get('days', 7), time_of_day_size=params.get('time_of_day_size', 24), dims=params.get('dims', 16), order=params.get('order', 2), normalization=params.get('normalization', 'batch'))
    )


SPEC = ModelSpec(
    name='STPGNN',
    module='models.stpgnn',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Spatio-Temporal Pivotal Graph Neural Networks for Traffic Flow Forecasting',
        venue='AAAI 2024',
        year=2024,
        url='',
    ),
    source=SourceRef(),
    evidence="unverified",
    config_path='configs/models/STPGNN.toml',
    model_card='src/models/stpgnn/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('marks',),
    deviations=(),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)
