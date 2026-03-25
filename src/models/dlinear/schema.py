from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    kernel_size: int = 25
    individual: bool = False
