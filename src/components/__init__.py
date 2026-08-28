"""Shared forecasting components with a lightweight lazy public surface."""

from importlib import import_module

from components.catalog import (
    COMPONENT_CATALOG,
    ComponentCatalog,
    ComponentMatch,
    ComponentSpec,
)


_LAZY_EXPORTS = {
    "ChannelWiseLinear": ("components.channel_wise_linear", "ChannelWiseLinear"),
    "DLinearBackbone": ("components.dlinear", "DLinearBackbone"),
    "DiffusionConv2d": ("components.diffusion_conv", "DiffusionConv2d"),
    "GaussianParameterHead": (
        "components.gaussian_parameter_head",
        "GaussianParameterHead",
    ),
    "MambaBlock": ("components.mamba", "MambaBlock"),
    "MambaResidualBlock": ("components.mamba", "MambaResidualBlock"),
    "PatchTSTBackbone": ("components.patchtst", "PatchTSTBackbone"),
    "QuantileHead": ("components.quantile_head", "QuantileHead"),
    "RMSNorm": ("components.mamba", "RMSNorm"),
    "RevIN": ("components.revin", "RevIN"),
    "SoftDecisionTree": ("components.soft_tree", "SoftDecisionTree"),
    "SoftObliviousTree": ("components.soft_tree", "SoftObliviousTree"),
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
    "ChannelWiseLinear",
    "DLinearBackbone",
    "DiffusionConv2d",
    "GaussianParameterHead",
    "MambaBlock",
    "MambaResidualBlock",
    "PatchTSTBackbone",
    "QuantileHead",
    "RMSNorm",
    "RevIN",
    "SoftDecisionTree",
    "SoftObliviousTree",
]
