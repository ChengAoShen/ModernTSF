from pydantic import BaseModel, Field


class EvaluationConfig(BaseModel):
    metrics: list[str] = Field(
        default_factory=lambda: ["mae", "mse", "rmse", "mape", "mspe"]
    )
    enable_profile: bool = False
