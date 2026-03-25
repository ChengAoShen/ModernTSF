from pydantic import BaseModel

from benchmark.config.schema.dataset import DatasetConfig
from benchmark.config.schema.evaluation import EvaluationConfig
from benchmark.config.schema.model import ModelConfig
from benchmark.config.schema.runtime import ExperimentConfig
from benchmark.config.schema.task import TaskConfig
from benchmark.config.schema.training import TrainConfig


class RootConfig(BaseModel):
    experiment: ExperimentConfig
    dataset: DatasetConfig
    task: TaskConfig
    training: TrainConfig
    model: ModelConfig
    evaluation: EvaluationConfig = EvaluationConfig()
