"""Model specification for GWNet."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.gwnet.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated GWNet parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset
    (see ``run_one.py``) and are therefore not declared here.
    """

    enc_in: int
    input_dim: int = 3
    dropout: float = 0.3
    residual_channels: int = 16
    dilation_channels: int = 16
    skip_channels: int = 64
    end_channels: int = 128
    kernel_size: int = 2
    blocks: int = 2
    layers: int = 2


def build_model(cfg, params):
    """Construct GWNet from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, num_nodes=params.get('num_nodes', params['enc_in']), adj_mx=params.get('adj_mx'), input_dim=params.get('input_dim', 3), dropout=params.get('dropout', 0.3), residual_channels=params.get('residual_channels', 16), dilation_channels=params.get('dilation_channels', 16), skip_channels=params.get('skip_channels', 64), end_channels=params.get('end_channels', 128), kernel_size=params.get('kernel_size', 2), blocks=params.get('blocks', 2), layers=params.get('layers', 2))
    )


SPEC = ModelSpec(
    name='GWNet',
    module='models.gwnet',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Graph WaveNet for Deep Spatial-Temporal Graph Modeling',
        venue='IJCAI 2019',
        year=2019,
        url='https://www.ijcai.org/proceedings/2019/264',
    ),
    source=SourceRef(
        url='https://github.com/GestaltCogTeam/BasicTS',
        revision='c218c07b6ce5e4cf908b147fd180c486346fed9c',
        license='Apache-2.0',
    ),
    evidence="upstream-port",
    config_path='configs/models/GWNet.toml',
    model_card='src/models/gwnet/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('marks',),
    deviations=(
        'ModernTSF reconstructs BasicTS history_data from the common value/mark signature.',
        'Dataset adjacency is converted to the official forward/reverse random-walk supports; adaptive adjacency remains enabled.',
        'The common runner objective replaces the official masked-MAE training objective.',
        'The display preset uses reduced channel widths and fewer blocks than the paper experiments.',
    ),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)
