"""Dataset parameter schemas."""

from data.schemas.datasets.custom import DatasetParameterConfig as CustomConfig
from data.schemas.datasets.ett import DatasetParameterConfig as ETTConfig
from data.schemas.datasets.periodic import DatasetParameterConfig as PeriodicConfig
from data.schemas.datasets.solar import DatasetParameterConfig as SolarConfig
from data.schemas.datasets.trend import DatasetParameterConfig as TrendConfig

__all__ = [
    "CustomConfig",
    "ETTConfig",
    "PeriodicConfig",
    "SolarConfig",
    "TrendConfig",
]
