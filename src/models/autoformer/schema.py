from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    dec_in: int
    c_out: int
    freq: str = "h"
    embed: str = "timeF"
    d_model: int = 512
    n_heads: int = 8
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 2048
    moving_avg: int = 25
    factor: int = 1
    dropout: float = 0.1
    num_class: int = 7
    activation: str = "gelu"
    task_name: str = "long_term_forecast"
