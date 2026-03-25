from pydantic import BaseModel, ConfigDict, Field


class ExperimentRuntimeConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    device: str = "cuda"
    use_multi_gpu: bool = False
    device_ids: list[int] = Field(default_factory=lambda: [0], alias="gpus")
    amp: bool = False
    num_workers: int = 4


class ExperimentConfig(BaseModel):
    description: str
    random_seed: int
    work_dir: str = "./work_dirs"
    runtime: ExperimentRuntimeConfig = ExperimentRuntimeConfig()
