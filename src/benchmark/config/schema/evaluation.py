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
    # Canonical quantile levels (the single source of truth for probabilistic
    # forecasting). Threaded into the evaluator and, for the "quantile" loss
    # only, injected into ``training.loss_params`` by ``run_one``. Ignored by
    # point models, so existing runs are unaffected.
    quantile_levels: list[float] = Field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    )
    enable_profile: bool = False
    # Evaluation strategy. "fixed" (default) keeps the historical fixed-window
    # evaluation untouched. "rolling" opts into a TFB-style rolling forecast.
    strategy: Literal["fixed", "rolling"] = "fixed"
    rolling: RollingConfig = Field(default_factory=RollingConfig)
