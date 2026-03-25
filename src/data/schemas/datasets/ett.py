from pydantic import BaseModel, Field


class DatasetParameterConfig(BaseModel):
    target: str
    scale: bool = True
    split_ratio: list[float] = Field(default_factory=lambda: [12.0, 4.0, 4.0])
