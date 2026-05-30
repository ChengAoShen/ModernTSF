from pydantic import BaseModel


class ModelParameterConfig(BaseModel):
    enc_in: int
    stack_types: list[str] = ["identity", "identity", "identity"]
    n_blocks: list[int] = [1, 1, 1]
    mlp_units: list = [[256, 256]]
    n_pool_kernel_size: list[int] = [2, 2, 1]
    n_freq_downsample: list[int] = [4, 2, 1]
    pooling_mode: str = "MaxPool1d"
    interpolation_mode: str = "linear"
    dropout: float = 0.0
    activation: str = "ReLU"
    use_norm: bool = True
