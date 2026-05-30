from typing import Literal

from pydantic import BaseModel


class TaskConfig(BaseModel):
    """Task configuration.

    All modes perform forecasting; ``mode`` selects the *data setting* (how a
    batch is shaped and what the model receives).

    .. note::

        ``mode`` is **advisory / declarative**: it documents the intended data
        setting and is validated here, but no runtime branch reads it. The
        actual batch shaping comes from the dataset (node-structured datasets
        pack the value into the series slot and covariates into a 4-D stamp
        slot) and from each model adapter (which reshapes via
        :mod:`models._external.marks`, polymorphic on mark rank). Setting a
        ``mode`` incompatible with the chosen model/dataset is therefore not
        rejected automatically — pick a combination supported by the model
        (see ``docs/en/task-modes.md``).

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
        * ``"covariate"`` — like ``"spatiotemporal"`` but the model also
          receives the *future* (known) covariate block ``(B, pred_len, N, F)``
          on the decoder side. Used e.g. by the air-quality forecasters
          CauAir / AirCade, which consume future weather covariates.

    seq_len, label_len, pred_len : int
        Window lengths.
    features : str
        Feature mode ("M", "S", "MS"); only used by ``time_series``.
    inverse : bool
        Whether to inverse-transform predictions before metrics.
    """

    mode: Literal["time_series", "spatiotemporal", "covariate"] = "time_series"
    seq_len: int
    label_len: int
    pred_len: int
    features: str = "M"
    inverse: bool = False
