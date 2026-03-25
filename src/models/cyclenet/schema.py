from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    cycle: int = 24
    model_type: str = "linear"
    d_model: int = 512
    use_revin: bool = True
