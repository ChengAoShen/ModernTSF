"""Model specification for BigST."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.bigst.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated BigST parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset
    and are not declared here.
    """

    enc_in: int  # number of nodes N (required)
    input_dim: int = 3  # 1 value + calendar covariates [tod, dow]
    hid_dim: int = 16
    node_dim: int = 8
    time_dim: int = 8
    tod_size: int = 24
    dow_size: int = 7
    tau: float = 1.0
    random_feature_dim: int = 16
    dropout: float = 0.1
    use_residual: bool = True
    use_bn: bool = True


def build_model(cfg, params):
    """Construct BigST from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, num_nodes=params.get('num_nodes', params['enc_in']), adj_mx=params.get('adj_mx'), input_dim=params.get('input_dim', 3), hid_dim=params.get('hid_dim', 16), node_dim=params.get('node_dim', 8), time_dim=params.get('time_dim', 8), tod_size=params.get('tod_size', 24), dow_size=params.get('dow_size', 7), tau=params.get('tau', 1.0), random_feature_dim=params.get('random_feature_dim', 16), dropout=params.get('dropout', 0.1), use_residual=params.get('use_residual', True), use_bn=params.get('use_bn', True))
    )


SPEC = ModelSpec(
    name='BigST',
    module='models.bigst',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='BigST: Linear Complexity Spatio-Temporal Graph Neural Network for Traffic Forecasting on Large-Scale Road Networks',
        venue='PVLDB 2024',
        year=2024,
        url='https://www.vldb.org/pvldb/vol17/p1081-han.pdf',
    ),
    source=SourceRef(
        url='https://github.com/GestaltCogTeam/BasicTS',
        revision='c218c07b6ce5e4cf908b147fd180c486346fed9c',
        license='Apache-2.0',
    ),
    evidence="adaptation",
    config_path='configs/models/BigST.toml',
    model_card='src/models/bigst/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('marks',),
    deviations=(
        'Ports the BasicTS single-stage use_long=False path and omits the separately pretrained long-sequence feature extractor.',
        'Disables the optional spatial regularization loss and uses the repository runner loss instead of the official masked-MAE recipe.',
        'Generalizes the hidden, node, and time embedding widths and normalizes calendar indices for the shared input contract.',
    ),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)
