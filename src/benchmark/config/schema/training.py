from pydantic import BaseModel, Field


class TrainOptimizerConfig(BaseModel):
    name: str = "Adam"
    lr: float = 0.001
    weight_decay: float = 0.0001
    lradj: str = "constant"
    params: dict = Field(default_factory=dict)


class TrainCheckpointConfig(BaseModel):
    strategy: str = "best"
    save_k: int = 3


class TrainConfig(BaseModel):
    epochs: int
    batch_size: int = 32
    loss: str = "mse"
    loss_params: dict = Field(default_factory=dict)
    patience: int = 3
    optimizer: TrainOptimizerConfig = TrainOptimizerConfig()
    checkpoint: TrainCheckpointConfig = TrainCheckpointConfig()
