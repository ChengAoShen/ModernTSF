from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    c_out: int | None = None
    d_model: int = 16
    e_layers: int = 1
    down_sampling_window: int = 2
    down_sampling_layers: int = 1
    begin_order: int = 0
    moving_avg: int = 25
    dropout: float = 0.1
    embed: str = "timeF"
    freq: str = "h"
    use_norm: int = 1
