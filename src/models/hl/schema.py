"""Parameter schema for the HL baseline."""

from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    """HL has no tunable parameters beyond enc_in (num nodes)."""

    enc_in: int = 207
