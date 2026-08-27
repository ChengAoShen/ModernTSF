"""Shared forecasting components with a lightweight lazy public surface."""

from importlib import import_module

from components.catalog import (
    COMPONENT_CATALOG,
    ComponentCatalog,
    ComponentMatch,
    ComponentSpec,
)


_LAZY_EXPORTS = {
    "DLinearBackbone": ("components.dlinear", "DLinearBackbone"),
    "MambaBlock": ("components.mamba", "MambaBlock"),
    "MambaResidualBlock": ("components.mamba", "MambaResidualBlock"),
    "PatchTSTBackbone": ("components.patchtst", "PatchTSTBackbone"),
    "QuantileHead": ("components.quantile_head", "QuantileHead"),
    "RMSNorm": ("components.mamba", "RMSNorm"),
    "RevIN": ("components.revin", "RevIN"),
}


def __getattr__(name: str):
    """Load optional torch-backed symbols only when a caller requests one."""
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, symbol_name = target
    value = getattr(import_module(module_name), symbol_name)
    globals()[name] = value
    return value

__all__ = [
    "COMPONENT_CATALOG",
    "ComponentCatalog",
    "ComponentMatch",
    "ComponentSpec",
    "DLinearBackbone",
    "MambaBlock",
    "MambaResidualBlock",
    "PatchTSTBackbone",
    "QuantileHead",
    "RMSNorm",
    "RevIN",
]
