from pydantic import BaseModel


class DatasetConfig(BaseModel):
    name: str
    alias: str | None = None
    root_path: str = "./data/"
    data_path: str
    params: dict
