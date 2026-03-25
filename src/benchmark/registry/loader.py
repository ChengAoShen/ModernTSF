"""Registry bootstrap helpers."""

from __future__ import annotations

from benchmark.registry.datasets import register_dataset_by_name
from benchmark.registry.metrics import register_metric_by_name
from benchmark.registry.models import register_model_by_name


def register_from_config(config) -> None:
    """Register datasets, models, and metrics referenced by a config.

    Parameters
    ----------
    config : RootConfig
        Validated config object.

    Returns
    -------
    None
    """
    register_dataset_by_name(config.dataset.name)
    register_model_by_name(config.model.name)
    if config.evaluation.metrics:
        for metric_name in config.evaluation.metrics:
            register_metric_by_name(metric_name)
