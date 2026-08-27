"""Model specification for STTN."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.sttn.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """Validated STTN parameters supplied via ``model.params``."""

    enc_in: int
    cov_dim: int = 2
    d_model: int = 64
    mlp_expand: int = 4
    num_layers: int = 3
    dropout: float = 0.1
    adj_type: str = "doubletransition"


def build_model(cfg, params):
    """Construct STTN from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], adj_mx=params.get('adj_mx'), cov_dim=params.get('cov_dim', 2), d_model=params.get('d_model', 64), mlp_expand=params.get('mlp_expand', 4), num_layers=params.get('num_layers', 3), dropout=params.get('dropout', 0.1), adj_type=params.get('adj_type', 'doubletransition'))
    )


SPEC = ModelSpec(
    name='STTN',
    module='models.sttn',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='Spatial-Temporal Transformer Networks for Traffic Flow Forecasting',
        venue='arXiv preprint',
        year=2020,
        url='https://arxiv.org/abs/2001.02908',
    ),
    source=SourceRef(
        url='https://github.com/xumingxingsjtu/STTN',
        revision='d24f8d331a6d81b819cfe0a9430793ae028d25ad',
        license='NOASSERTION',
    ),
    evidence="unverified",
    config_path='configs/models/STTN.toml',
    model_card='src/models/sttn/README.md',
    smoke_config=None,
    capabilities=frozenset(['spatiotemporal']),
    components=('graph_utils', 'marks'),
    deviations=(
        'The local PyTorch core came through CauAir revision 73dae00ca6ad14abb15174a0a0286d500e868b94 and is not a direct port of the pinned official TensorFlow repository.',
        'Unlike the paper implementation, the local spatial block mixes learned attention with fixed-adjacency second-order graph convolution and the adapter creates a dense graph when none is supplied.',
        'The official repository uses TensorFlow transformer layers and paper-specific spatial/temporal encodings; parameterization, initialization, preprocessing, and output head differ from this PyTorch baseline.',
        'Attention heads are fixed to four in the local core; mlp_expand controls the feed-forward width and replaces the previously misleading n_heads public parameter.',
        'Raw calendar marks are appended as covariates, which is not the official PeMS preprocessing contract.',
        'Neither the pinned official repository nor the CauAir source declares a code license, and no numerical parity is available.',
    ),
    contract_task={'seq_len': 12, 'pred_len': 12, 'label_len': 0},
)
