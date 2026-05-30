from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    d_model: int = 64
    n_heads: int = 4
    e_layers: int = 2
    d_ff: int = 64
    dropout: float = 0.1
    fc_dropout: float = 0.1
    factor: int = 3
    activation: str = "gelu"
    moving_avg: int = 25
    num_experts: int = 4
    k: int = 2
    hidden_size: int = 64
    noisy_gating: bool = True
    CI: bool = True
    output_attention: bool = False
