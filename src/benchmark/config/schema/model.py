from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: str
    params: dict = Field(default_factory=dict)
