from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    period: int = 24
    d_model: int = 64
    model_type: str = "linear"
