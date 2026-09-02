"""Thin interfaces for official time-series foundation-model runtimes."""

from models._foundation.official import (
    ChronosRuntime,
    MoiraiRuntime,
    TimesFMRuntime,
)
from models._foundation.runtime import (
    FoundationDependencyError,
    FoundationForecast,
    FoundationModel,
    FoundationRuntime,
    FoundationSource,
)

__all__ = [
    "ChronosRuntime",
    "FoundationDependencyError",
    "FoundationForecast",
    "FoundationModel",
    "FoundationRuntime",
    "FoundationSource",
    "MoiraiRuntime",
    "TimesFMRuntime",
]
