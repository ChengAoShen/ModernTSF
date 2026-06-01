"""Helpers for writing CSV summaries and sweep metadata."""

from __future__ import annotations

import csv
import os
import json
from typing import Iterable


def _read_existing_header(path: str) -> list[str]:
    """Return the header row already written to a CSV file (``[]`` if empty)."""
    with open(path, "r", newline="") as f:
        try:
            return next(csv.reader(f))
        except StopIteration:
            return []


def write_csv_summary(
    path: str,
    row: dict,
    header: Iterable[str] | None = None,
) -> None:
    """Append a single summary row to a CSV file.

    When ``header`` is ``None`` the column set is derived from the rows. Because
    different runs can emit different metric sets (e.g. a graph model adds
    ``corr``/``rse``/``wape``/``smape`` + timing columns that a plain model
    omits), a later row whose keys are a *superset* of the on-disk header
    triggers a **header migration**: the file is rewritten with the union header
    and the older rows are back-filled with empty cells, then atomically
    replaced. This keeps every value under its named column. Subset rows are
    appended under the existing header (missing cells left blank). Passing an
    explicit ``header`` keeps a fixed schema and tolerates missing/extra keys
    without column drift.

    (Previously the header was frozen at first write while every later row was
    written in its own key order — heterogeneous metric sets silently shifted
    columns, corrupting ``performance.csv`` for downstream consumers.)

    Parameters
    ----------
    path : str
        Output CSV path.
    row : dict
        Row content.
    header : Iterable[str] | None, optional
        Field names for the CSV header. When provided, a fixed schema is used.

    Returns
    -------
    None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # New file: write the header (explicit or derived from the row) + the row.
    if not os.path.exists(path):
        fieldnames = list(header) if header is not None else list(row.keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=fieldnames, restval="", extrasaction="ignore"
            )
            writer.writeheader()
            writer.writerow(row)
        return

    # Existing file with a caller-supplied fixed header (e.g. profile.csv):
    # append, tolerating missing/extra keys without shifting columns.
    if header is not None:
        with open(path, "a", newline="") as f:
            csv.DictWriter(
                f, fieldnames=list(header), restval="", extrasaction="ignore"
            ).writerow(row)
        return

    # Existing file, header derived from rows: reconcile against this row's keys.
    existing = _read_existing_header(path)
    new_keys = [k for k in row.keys() if k not in existing]

    if not new_keys:
        with open(path, "a", newline="") as f:
            csv.DictWriter(
                f, fieldnames=existing, restval="", extrasaction="ignore"
            ).writerow(row)
        return

    # New columns appeared -> migrate to the union header (existing first, then
    # the new keys), back-filling prior rows, and atomically replace the file.
    union = existing + new_keys
    with open(path, "r", newline="") as f:
        old_rows = list(csv.DictReader(f))
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=union, restval="", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(old_rows)
        writer.writerow(row)
    os.replace(tmp_path, path)


def _flatten_params(params: dict, prefix: str = "") -> dict:
    """Flatten nested params into dot-delimited keys.

    Lists/tuples are JSON-encoded to preserve structure in CSV outputs.

    Parameters
    ----------
    params : dict
        Parameter dictionary.
    prefix : str, optional
        Prefix used during recursion.

    Returns
    -------
    dict
        Flattened parameter mapping.
    """
    flat = {}
    for key, value in params.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flat.update(_flatten_params(value, path))
        elif isinstance(value, (list, tuple)):
            flat[path] = json.dumps(value, ensure_ascii=True)
        else:
            flat[path] = value
    return flat


def _append_sweep_values(row: dict, raw: dict, sweep_keys: list[str]) -> None:
    """Append sweep values to the output row in place.

    Parameters
    ----------
    row : dict
        Summary row to update.
    raw : dict
        Raw expanded config dictionary.
    sweep_keys : list[str]
        Dot-delimited sweep keys to include.

    Returns
    -------
    None
    """
    if not sweep_keys:
        return
    flattened = _flatten_params(raw)
    for key in sweep_keys:
        if key in flattened:
            row[f"sweep.{key}"] = flattened[key]


def default_summary_row(
    base: dict,
    metrics: dict[str, float],
    raw: dict | None = None,
    sweep_keys: list[str] | None = None,
) -> dict:
    """Build a normalized summary row for CSV output.

    Parameters
    ----------
    base : dict
        Required metadata fields (dataset, model, lengths, seed, run_id).
    metrics : dict[str, float]
        Metric values to include.
    raw : dict | None, optional
        Raw expanded config for sweep metadata.
    sweep_keys : list[str] | None, optional
        Dot-delimited sweep keys to include.

    Returns
    -------
    dict
        Output row dictionary.
    """
    row = {
        "dataset": base.get("dataset"),
        "model": base.get("model"),
        "seq_len": base.get("seq_len"),
        "pred_len": base.get("pred_len"),
        "seed": base.get("seed"),
        "run_id": base.get("run_id"),
    }

    metric_order = ["mae", "mse", "rmse", "mape", "mspe"]
    for name in metric_order:
        if name in metrics:
            row[name] = metrics[name]
    for name, value in metrics.items():
        if name not in row:
            row[name] = value

    # Timing columns. ``fit_time`` is the total training wall-clock; the
    # ``inference_time`` is the test-set evaluation wall-clock. Only emitted when
    # provided so existing callers/headers are unaffected.
    if "fit_time" in base:
        row["fit_time"] = base["fit_time"]
    if "inference_time" in base:
        row["inference_time"] = base["inference_time"]

    if raw and sweep_keys:
        _append_sweep_values(row, raw, sweep_keys)
    return row
