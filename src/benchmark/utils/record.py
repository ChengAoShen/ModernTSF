"""Emit a self-describing ``record.json`` (a ``tsf_core.RunRecord``) per run.

Each ModernTSF run writes one ``record.json`` next to its CSV outputs. Unlike
``performance.csv`` (positional, shared-header), this is self-describing and
schema-validated, so ``tsf submit`` and the TSEval leaderboard can ingest a run
with zero column-alignment guesswork. Writing is best-effort: any failure
degrades to a raw-dict fallback (or a logged warning) and never breaks training.
"""

from __future__ import annotations

import json
import os
import warnings
from datetime import datetime, timezone


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_record_dict(
    *,
    config,
    device,
    run_id: str,
    dataset_id: str,
    metrics: dict,
    fit_time: float | None,
    inference_time: float | None,
    n_windows: int | None = None,
    repo_root: str | None = None,
) -> dict:
    """Assemble + validate a RunRecord, returning its JSON-able dict.

    Raises on validation failure; callers should catch and fall back.
    """
    from tsf_core import (
        HorizonResult,
        MetricSet,
        RunConfigRef,
        RunEnv,
        RunRecord,
        RunTiming,
    )

    from benchmark.utils.env import collect_env, collect_git

    mode = config.task.mode
    # Track defaults to the task mode, but a dataset config may override it
    # (e.g. track = "realtime" for live datasets like stock_hs300).
    track = getattr(config.dataset, "track", None) or mode
    metric_keys = set(MetricSet.model_fields)
    metric_set = MetricSet(**{k: v for k, v in metrics.items() if k in metric_keys})

    horizon = HorizonResult(
        horizon=config.task.pred_len,
        metrics=metric_set,
        timing=RunTiming(fit_time_sec=fit_time, inference_time_sec=inference_time),
        n_windows=n_windows,
        run_id=run_id,
    )

    env_fields = set(RunEnv.model_fields)
    env_data = collect_env(device)
    env_data.update(collect_git(repo_root))
    env = RunEnv(**{k: v for k, v in env_data.items() if k in env_fields})

    record = RunRecord(
        record_id=(
            f"{config.model.name}__{dataset_id}__seed{config.experiment.random_seed}"
            f"__pl{config.task.pred_len}"
        ),
        model=config.model.name,
        dataset_id=dataset_id,
        track=track,
        mode=mode,
        seed=config.experiment.random_seed,
        results=[horizon],
        config=RunConfigRef(
            mode=mode,
            seq_len=config.task.seq_len,
            label_len=config.task.label_len,
            pred_len=config.task.pred_len,
            features=config.task.features,
        ),
        env=env,
        created_at=_iso_now(),
    )
    return record.model_dump(mode="json")


def write_run_record(record_path: str, **kwargs) -> None:
    """Build and write ``record.json``. Never raises.

    On a validation/build error, writes a minimal raw-dict fallback so the run
    still leaves a machine-readable artifact; on a write error, warns only.
    """
    try:
        data = build_record_dict(**kwargs)
    except Exception as exc:  # pragma: no cover - defensive
        warnings.warn(
            f"record.json: could not build a validated RunRecord ({exc!r}); "
            "writing a raw fallback."
        )
        config = kwargs.get("config")
        data = {
            "schema_version": "fallback",
            "run_id": kwargs.get("run_id"),
            "model": getattr(getattr(config, "model", None), "name", None),
            "dataset_id": kwargs.get("dataset_id"),
            "metrics": kwargs.get("metrics"),
            "fit_time_sec": kwargs.get("fit_time"),
            "inference_time_sec": kwargs.get("inference_time"),
            "created_at": _iso_now(),
        }
    try:
        os.makedirs(os.path.dirname(record_path), exist_ok=True)
        with open(record_path, "w") as f:
            json.dump(data, f, indent=2, default=str)
    except Exception as exc:  # pragma: no cover - defensive
        warnings.warn(f"record.json: write failed ({exc!r})")
