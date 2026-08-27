"""Model specification for CrossGNN."""

from __future__ import annotations

from benchmark.registry.models import ModelSpec, PaperRef, SourceRef
from models.crossgnn.model import Model

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    e_layers: int = 2
    anti_ood: bool = True
    tk: int = 10
    scale_number: int = 4
    use_tgcn: bool = True
    use_ngcn: bool = True
    dropout: float = 0.1
    tvechidden: int = 8
    nvechidden: int = 8
    hidden: int = 16


def build_model(cfg, params):
    """Construct CrossGNN from a validated run configuration."""
    return (
    Model(seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len, enc_in=params['enc_in'], e_layers=params.get('e_layers', 2), anti_ood=bool(params.get('anti_ood', True)), tk=params.get('tk', 10), scale_number=params.get('scale_number', 4), use_tgcn=bool(params.get('use_tgcn', True)), use_ngcn=bool(params.get('use_ngcn', True)), dropout=params.get('dropout', 0.1), tvechidden=params.get('tvechidden', 8), nvechidden=params.get('nvechidden', 8), hidden=params.get('hidden', 16))
    )


SPEC = ModelSpec(
    name='CrossGNN',
    module='models.crossgnn',
    model_class=Model,
    factory=build_model,
    params_schema=ModelParameterConfig,
    paper=PaperRef(
        title='CrossGNN: Confronting Noisy Multivariate Time Series Via Cross Interaction Refinement',
        venue='NeurIPS 2023',
        year=2023,
        url='https://proceedings.neurips.cc/paper_files/paper/2023/hash/9278abf072b58caf21d48dd670b4c721-Abstract-Conference.html',
    ),
    source=SourceRef(url='https://github.com/hqh0728/CrossGNN', revision='0407abd085ee8342abe0bbe6de5b2ab17c44373c', license='NOASSERTION'),
    evidence="unverified",
    config_path='configs/models/CrossGNN.toml',
    model_card='src/models/crossgnn/README.md',
    smoke_config=None,
    capabilities=frozenset(['time-series']),
    components=(),
    deviations=(
        'FFT-derived multi-scale pooling, cross-scale temporal graph refinement, cross-variable graph refinement, learned sparse adjacency, and anti-OOD last-value normalization were compared with the pinned author repository.',
        'ModernTSF removes hard-coded CUDA placement, fixes the source negative-adjacency row-threshold indexing, and removes permanently unreachable GraphforPre/layer-normalization parameters and the inert individual option.',
        'The author repository has no license covering CrossGNN itself; licenses in bundled baseline subdirectories do not apply. No checkpoint parity evidence is available, so verification remains blocked.',
    ),
    contract_task={'seq_len': 96, 'pred_len': 96, 'label_len': 0},
    contract_seeds=(0, 18, 24),
)
