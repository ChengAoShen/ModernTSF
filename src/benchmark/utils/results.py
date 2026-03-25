"""Helpers for writing CSV summaries and sweep metadata."""

from __future__ import annotations

import csv
import os
import json
from typing import Iterable


def write_csv_summary(
    path: str,
    row: dict,
    header: Iterable[str] | None = None,
) -> None:
    """Append a single summary row to a CSV file.

    Parameters
    ----------
    path : str
        Output CSV path.
    row : dict
        Row content.
    header : Iterable[str] | None, optional
        Field names for the CSV header.

    Returns
    -------
    None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)

    if header is None:
        header = list(row.keys())

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(header))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


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

    if raw and sweep_keys:
        _append_sweep_values(row, raw, sweep_keys)
    return row
