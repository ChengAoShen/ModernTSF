from benchmark.registry.datasets import DATASET_REGISTRY
from benchmark.registry.loader import register_from_config
from benchmark.registry.losses import LOSS_REGISTRY
from benchmark.registry.metrics import METRIC_REGISTRY
from benchmark.registry.models import MODEL_REGISTRY

__all__ = [
    "DATASET_REGISTRY",
    "LOSS_REGISTRY",
    "METRIC_REGISTRY",
    "MODEL_REGISTRY",
    "register_from_config",
]
