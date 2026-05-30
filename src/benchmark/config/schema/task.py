from typing import Literal

from pydantic import BaseModel


class TaskConfig(BaseModel):
    """Task configuration.

    All modes perform forecasting; ``mode`` selects the *data setting* (how a
    batch is shaped and what the model receives).

    Parameters
    ----------
    mode : str
        Forecasting data setting, one of:

        * ``"time_series"`` (default) — classic multivariate time-series
          forecasting. Batches are ``(B, T, C)`` value tensors; every channel
          is a target.
        * ``"spatiotemporal"`` — node-structured forecasting. Batches carry a
          ``(B, T, N, 1 + F)`` tensor where channel 0 is the value and the
          remaining ``F`` channels are per-node covariates / calendar features.
          Only the value channel of all ``N`` nodes is the target.
        * ``"air_quality"`` — like ``"spatiotemporal"`` but the model also
          receives the *future* covariate block ``(B, pred_len, N, F)``.

    seq_len, label_len, pred_len : int
        Window lengths.
    features : str
        Feature mode ("M", "S", "MS"); only used by ``time_series``.
    inverse : bool
        Whether to inverse-transform predictions before metrics.
    """

    mode: Literal["time_series", "spatiotemporal", "air_quality"] = "time_series"
    seq_len: int
    label_len: int
    pred_len: int
    features: str = "M"
    inverse: bool = False
