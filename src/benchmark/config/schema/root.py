"""Strict top-level schema joining every benchmark configuration section."""

from pydantic import BaseModel, ConfigDict

from benchmark.config.schema.dataset import DatasetConfig
from benchmark.config.schema.evaluation import EvaluationConfig
from benchmark.config.schema.model import ModelConfig
from benchmark.config.schema.runtime import ExperimentConfig
from benchmark.config.schema.task import TaskConfig
from benchmark.config.schema.training import TrainConfig


class RootConfig(BaseModel):
    # Catches a misspelled top-level section (e.g. `[trainnig]`) at
    # config-load time instead of it silently falling back to defaults. Only
    # Every structural section is strict. The model and dataset ``params``
    # mappings are validated after name resolution by their registered schema.
    model_config = ConfigDict(extra="forbid")

    experiment: ExperimentConfig
    dataset: DatasetConfig
    task: TaskConfig
    training: TrainConfig
    model: ModelConfig
    evaluation: EvaluationConfig = EvaluationConfig()
