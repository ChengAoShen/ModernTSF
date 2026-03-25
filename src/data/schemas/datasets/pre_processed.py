from pydantic import BaseModel


class PreProcessedParameterConfig(BaseModel):
    """No dataset-level params — all preprocessing is done by tool/pre_process.py."""
