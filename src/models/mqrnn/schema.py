from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    hidden_size: int = 64
    num_layers: int = 1
    decoder_hidden: int = 64
    dropout: float = 0.1
    quantile_levels: list[float] | None = None
