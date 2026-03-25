from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    c_out: int
    freq: str = "h"
    embed: str = "timeF"
    e_layers: int = 2
    d_model: int = 512
    d_ff: int = 2048
    down_sampling_window: int = 1
    down_sampling_layers: int = 0
    down_sampling_method: str | None = None
    channel_independence: bool = False
    moving_avg: int = 25
    top_k: int = 5
    dropout: float = 0.0
    use_norm: bool = True
    decomp_method: str = "moving_avg"
