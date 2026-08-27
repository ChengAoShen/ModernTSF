"""Parameter contract for pre-windowed NumPy forecasting datasets."""

from pydantic import BaseModel


class PreProcessedParameterConfig(BaseModel):
    """No dataset-level params; preprocessing is done by `tsf dataset prepare`."""
