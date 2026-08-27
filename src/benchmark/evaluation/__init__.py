"""Forecast metrics and model-resource profiling."""

from benchmark.evaluation.metrics import collect_metrics
from benchmark.evaluation.profile import profile_model

__all__ = ["collect_metrics", "profile_model"]
