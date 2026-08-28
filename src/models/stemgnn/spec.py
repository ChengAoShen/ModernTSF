"""Model specification for StemGNN."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec
from models.stemgnn.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated StemGNN parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset
    (not declared here). ``enc_in`` (= number of nodes ``N``) is required.
    """

    enc_in: int
    input_dim: int = 3
    multi_layer: int = 3
    dropout_rate: float = 0.5
    leaky_rate: float = 0.2


def build_model(cfg, params):
    """Construct StemGNN from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, num_nodes=params.get('num_nodes', params['enc_in']), adj_mx=params.get('adj_mx'), input_dim=params.get('input_dim', 3), multi_layer=params.get('multi_layer', 3), dropout_rate=params.get('dropout_rate', 0.5), leaky_rate=params.get('leaky_rate', 0.2))
    )


SPEC = ModelSpec(
    name='StemGNN',
    module='models.stemgnn',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/StemGNN.toml',
    model_card='src/models/stemgnn/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=(),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)
