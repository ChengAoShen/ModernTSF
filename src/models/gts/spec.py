"""Model specification for GTS."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec
from models.gts.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated GTS parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset and
    need not be declared in the TOML. ``enc_in`` (= number of nodes ``N``) is the
    only required field and serves as the fallback for ``num_nodes``.
    """

    enc_in: int
    input_dim: int = 3
    rnn_units: int = 16
    num_rnn_layers: int = 1
    max_diffusion_step: int = 2
    embedding_dim: int = 16
    node_feats_len: int = 40
    k: int = 3
    temp: float = 0.5


def build_model(cfg, params):
    """Construct GTS from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, num_nodes=params.get('num_nodes', params['enc_in']), adj_mx=params.get('adj_mx'), input_dim=params.get('input_dim', 3), rnn_units=params.get('rnn_units', 16), num_rnn_layers=params.get('num_rnn_layers', 1), max_diffusion_step=params.get('max_diffusion_step', 2), embedding_dim=params.get('embedding_dim', 16), node_feats_len=params.get('node_feats_len', 40), k=params.get('k', 3), temp=params.get('temp', 0.5))
    )


SPEC = ModelSpec(
    name='GTS',
    module='models.gts',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    config_path='configs/models/GTS.toml',
    model_card='src/models/gts/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('marks',),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)
