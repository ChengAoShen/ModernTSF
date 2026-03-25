from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    c_out: int
    freq: str = "h"
    embed: str = "timeF"
    d_model: int = 512
    e_layers: int = 2
    d_ff: int = 2048
    dropout: float = 0.1
    top_k: int = 5
    num_kernels: int = 6
    num_class: int = 7
    task_name: str = "long_term_forecast"
