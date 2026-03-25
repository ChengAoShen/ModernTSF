from pydantic import BaseModel


class TaskConfig(BaseModel):
    seq_len: int
    label_len: int
    pred_len: int
    features: str = "M"
    inverse: bool = False
