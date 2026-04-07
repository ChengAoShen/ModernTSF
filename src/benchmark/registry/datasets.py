"""Dataset registry and dynamic registration helpers."""

from __future__ import annotations

import importlib
from typing import Type

from pydantic import BaseModel


class DatasetRegistry:
    """Registry mapping dataset names to dataset classes and schemas."""

    def __init__(self) -> None:
        self._datasets: dict[str, tuple[Type, Type[BaseModel] | None]] = {}

    def register(
        self, name: str, dataset_cls: Type, schema: Type[BaseModel] | None = None
    ) -> None:
        """Register a dataset class with an optional parameter schema."""
        self._datasets[name] = (dataset_cls, schema)

    def get(self, name: str) -> tuple[Type, Type[BaseModel] | None]:
        """Get a dataset class and schema by name.

        Raises
        ------
        KeyError
            If the dataset is not registered.
        """
        if name not in self._datasets:
            raise KeyError(f"Dataset '{name}' is not registered")
        return self._datasets[name]

    def names(self) -> list[str]:
        """Return registered dataset names."""
        return sorted(self._datasets.keys())


DATASET_REGISTRY = DatasetRegistry()

DATASET_NAME_MAP = {
    "ETTh1": "data.datasets.ett",
    "ETTh2": "data.datasets.ett",
    "ETTm1": "data.datasets.ett",
    "ETTm2": "data.datasets.ett",
    "traffic": "data.datasets.custom",
    "weather": "data.datasets.custom",
    "electricity": "data.datasets.custom",
    "solar": "data.datasets.solar",
    "periodic": "data.datasets.periodic_data",
    "trend": "data.datasets.trend_data",
    "pre_processed": "data.datasets.pre_processed",
    "gift_eval": "data.datasets.gift_eval",
}

_REGISTERED_DATASETS: set[str] = set()


def register_dataset_by_name(name: str) -> None:
    """Import and register a dataset using the name map.

    Parameters
    ----------
    name : str
        Dataset name from the config.

    Returns
    -------
    None

    Raises
    ------
    KeyError
        If the dataset name is not mapped.
    ModuleNotFoundError
        If the mapped module cannot be imported.
    AttributeError
        If the module has no register() function.
    """
    if name in _REGISTERED_DATASETS:
        return
    module_name = DATASET_NAME_MAP.get(name)
    if module_name is None:
        available = ", ".join(sorted(DATASET_NAME_MAP.keys())) or "<none>"
        raise KeyError(
            f"Dataset '{name}' is not mapped. Update DATASET_NAME_MAP in "
            f"benchmark.registry.datasets. Available: {available}"
        )
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise ModuleNotFoundError(
                f"Dataset registry module not found: {module_name}. "
                "Expected module path in DATASET_NAME_MAP"
            ) from exc
        raise ImportError(
            f"Failed to import '{module_name}' due to missing dependency: {exc}"
        ) from exc

    register_fn = getattr(module, "register", None)
    if register_fn is None:
        raise AttributeError(
            f"Dataset registry '{module_name}' must define a register() function"
        )
    register_fn()
    _REGISTERED_DATASETS.add(name)
