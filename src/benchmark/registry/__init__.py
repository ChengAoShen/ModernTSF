"""Flat runtime catalogs for datasets, losses, metrics, and model specifications."""

from benchmark.registry.datasets import DATASET_REGISTRY
from benchmark.registry.loader import register_from_config
from benchmark.registry.losses import LOSS_REGISTRY
from benchmark.registry.metrics import METRIC_REGISTRY
from benchmark.registry.models import MODEL_CATALOG

__all__ = [
    "DATASET_REGISTRY",
    "LOSS_REGISTRY",
    "METRIC_REGISTRY",
    "MODEL_CATALOG",
    "register_from_config",
]
