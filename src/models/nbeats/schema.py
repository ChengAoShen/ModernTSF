from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    stack_types: list[str] = ["trend", "seasonality", "generic"]
    nb_blocks_per_stack: int = 3
    thetas_dim: list[int] = [4, 8, 8]
    hidden_layer_units: int = 256
    share_weights_in_stack: bool = False
    nb_harmonics: int | None = None
