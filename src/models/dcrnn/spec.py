"""Model specification for DCRNN."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.dcrnn.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated DCRNN parameters supplied via ``model.params``.

    ``num_nodes`` and ``adj_mx`` are injected by the runner from the dataset
    (see ``run_one.py``) and are therefore not declared here.
    """

    enc_in: int
    input_dim: int = 3
    rnn_units: int = 16
    num_rnn_layers: int = 1
    max_diffusion_step: int = 2


def build_model(cfg, params):
    """Construct DCRNN from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, num_nodes=params.get('num_nodes', params['enc_in']), adj_mx=params.get('adj_mx'), input_dim=params.get('input_dim', 3), rnn_units=params.get('rnn_units', 16), num_rnn_layers=params.get('num_rnn_layers', 1), max_diffusion_step=params.get('max_diffusion_step', 2))
    )


SPEC = ModelSpec(
    name='DCRNN',
    module='models.dcrnn',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Diffusion Convolutional Recurrent Neural Network: Data-Driven Traffic Forecasting',
        venue='ICLR 2018',
        year=2018,
        url='https://openreview.net/forum?id=SJiHXGWAZ',
    ),
    source=SourceRef(
        url='https://github.com/liyaguang/DCRNN',
        revision='602afd9d767d3aa1c9b3eac51710d6aeee12c227',
        license='MIT',
    ),
    evidence="unverified",
    config_path='configs/models/DCRNN.toml',
    model_card='src/models/dcrnn/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('marks',),
    deviations=(
        'The in-tree PyTorch implementation is adapted from BasicTS rather than ported from the pinned official TensorFlow repository; the exact BasicTS source revision was not recorded.',
        'The preset uses 16 hidden units and one recurrent layer instead of the official METR-LA preset with 64 units and two layers.',
        'The adapter uses value plus two normalized calendar channels, while the official METR-LA preset uses input_dim=2.',
        'Scheduled sampling is disabled because the generic forecaster call does not pass future targets into the DCRNN decoder.',
        'Official graph construction, normalization, masked MAE training, and data preprocessing are not reproduced by the model package.',
    ),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)
