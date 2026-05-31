from typing import Literal, Optional

from pydantic import BaseModel, Field


class RollingConfig(BaseModel):
    """Optional rolling-forecast sub-config.

    Only consulted when ``EvaluationConfig.strategy == "rolling"``. All fields
    are optional; the defaults keep the rolling walk as permissive as possible
    (predict ``pred_len``, advance by 1 step, continue until the test data is
    exhausted).
    """

    horizon: Optional[int] = None
    stride: int = 1
    num_rollings: Optional[int] = None


class EvaluationConfig(BaseModel):
    metrics: list[str] = Field(
        default_factory=lambda: ["mae", "mse", "rmse", "mape", "mspe"]
    )
    enable_profile: bool = False
    # Evaluation strategy. "fixed" (default) keeps the historical fixed-window
    # evaluation untouched. "rolling" opts into a TFB-style rolling forecast.
    strategy: Literal["fixed", "rolling"] = "fixed"
    rolling: RollingConfig = Field(default_factory=RollingConfig)
