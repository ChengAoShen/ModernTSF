"""Aggregate performance and profile CSVs for a dataset."""

from __future__ import annotations

import argparse
import csv
import glob
import os
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class FilterCondition:
    key: str
    op: str
    value: str


DEFAULT_PERF_FIELDS = "model,seq_len,pred_len,mse,mae"
DEFAULT_PROF_FIELDS = "latency_avg_ms,throughput_samples_sec,total_params,peak_vram_mb"


def _parse_filters(filter_expr: str | None) -> list[FilterCondition]:
    if not filter_expr:
        return []
    conditions: list[FilterCondition] = []
    for raw in filter_expr.split(","):
        token = raw.strip()
        if not token:
            continue
        op = None
        for candidate in ("<=", ">=", "!=", "=", "<", ">", "~"):
            if candidate in token:
                op = candidate
                break
        if op is None:
            raise ValueError(f"Invalid filter token: {token}")
        key, value = token.split(op, 1)
        key = key.strip()
        value = value.strip()
        if not key:
            raise ValueError(f"Invalid filter token: {token}")
        conditions.append(FilterCondition(key=key, op=op, value=value))
    return conditions


def _to_number(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _match_condition(row: dict[str, str], condition: FilterCondition) -> bool:
    if condition.key not in row:
        return False
    raw_value = row.get(condition.key, "")
    if condition.op == "~":
        return condition.value in raw_value

    left_num = _to_number(raw_value)
    right_num = _to_number(condition.value)
    if left_num is not None and right_num is not None:
        if condition.op == "=":
            return left_num == right_num
        if condition.op == "!=":
            return left_num != right_num
        if condition.op == "<":
            return left_num < right_num
        if condition.op == ">":
            return left_num > right_num
        if condition.op == "<=":
            return left_num <= right_num
        if condition.op == ">=":
            return left_num >= right_num
        return False

    if condition.op == "=":
        return raw_value == condition.value
    if condition.op == "!=":
        return raw_value != condition.value
    if condition.op == "<":
        return raw_value < condition.value
    if condition.op == ">":
        return raw_value > condition.value
    if condition.op == "<=":
        return raw_value <= condition.value
    if condition.op == ">=":
        return raw_value >= condition.value
    return False


def _filter_rows(
    rows: Iterable[dict[str, str]], conditions: list[FilterCondition]
) -> list[dict[str, str]]:
    if not conditions:
        return list(rows)
    result = []
    for row in rows:
        if all(_match_condition(row, cond) for cond in conditions):
            result.append(row)
    return result


def _collect_csv_files(dataset_dir: str, filename: str) -> list[str]:
    pattern = os.path.join(dataset_dir, "*", filename)
    return sorted(glob.glob(pattern))


def _read_csvs(paths: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    fieldnames: list[str] = []
    rows: list[dict[str, str]] = []
    for path in paths:
        with open(path, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                continue
            for name in reader.fieldnames:
                if name not in fieldnames:
                    fieldnames.append(name)
            for row in reader:
                rows.append({k: ("" if v is None else str(v)) for k, v in row.items()})
    return fieldnames, rows


def _write_csv(path: str, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _parse_fields(fields_expr: str | None, fieldnames: list[str]) -> list[str]:
    if fields_expr is None:
        return fieldnames
    tokens = [token.strip() for token in fields_expr.split(",") if token.strip()]
    if not tokens:
        return fieldnames
    selected = [name for name in tokens if name in fieldnames]
    missing = [name for name in tokens if name not in fieldnames]
    if missing:
        print(f"Warning: missing fields ignored: {', '.join(missing)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate performance and profile CSVs for a dataset"
    )
    parser.add_argument("--dataset", required=True, type=str, help="Dataset name")
    parser.add_argument(
        "--work-dir",
        type=str,
        default="./work_dirs",
        help="Root work directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV path (default: work_dirs/<dataset>/results_all.csv)",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Filter expression, e.g. 'pred_len=96,model~Linear'",
    )
    parser.add_argument(
        "--perf-fields",
        type=str,
        default=DEFAULT_PERF_FIELDS,
        help="Comma-separated fields to keep from performance.csv",
    )
    parser.add_argument(
        "--prof-fields",
        type=str,
        default=DEFAULT_PROF_FIELDS,
        help="Comma-separated fields to keep from profile.csv",
    )
    args = parser.parse_args()

    dataset_dir = os.path.join(args.work_dir, args.dataset)
    perf_paths = _collect_csv_files(dataset_dir, "performance.csv")
    prof_paths = _collect_csv_files(dataset_dir, "profile.csv")

    if not perf_paths and not prof_paths:
        raise SystemExit(f"No performance.csv or profile.csv found under {dataset_dir}")

    perf_fieldnames: list[str] = []
    perf_rows: list[dict[str, str]] = []
    if perf_paths:
        perf_fieldnames, perf_rows = _read_csvs(perf_paths)

    prof_fieldnames: list[str] = []
    prof_rows: list[dict[str, str]] = []
    if prof_paths:
        prof_fieldnames, prof_rows = _read_csvs(prof_paths)

    perf_fields = _parse_fields(args.perf_fields, perf_fieldnames)
    prof_fields = _parse_fields(args.prof_fields, prof_fieldnames)
    if perf_paths and not perf_fields:
        raise SystemExit("No valid performance fields selected for output")
    if prof_paths and not prof_fields:
        raise SystemExit("No valid profile fields selected for output")

    prof_by_run: dict[str, dict[str, str]] = {}
    if prof_rows:
        for row in prof_rows:
            run_id = row.get("run_id", "")
            if run_id:
                prof_by_run[run_id] = row

    merged_rows: list[dict[str, str]] = []
    if perf_rows:
        for row in perf_rows:
            merged: dict[str, str] = {}
            for name in perf_fields:
                merged[name] = row.get(name, "")
            if prof_rows:
                prof_row = prof_by_run.get(row.get("run_id", ""))
                if prof_row:
                    for name in prof_fields:
                        merged[name] = prof_row.get(name, "")
                else:
                    for name in prof_fields:
                        merged[name] = ""
            merged_rows.append(merged)
    else:
        for row in prof_rows:
            merged: dict[str, str] = {}
            for name in prof_fields:
                merged[name] = row.get(name, "")
            merged_rows.append(merged)

    conditions = _parse_filters(args.filter)
    filtered_rows = _filter_rows(merged_rows, conditions)

    output_path = args.output
    if output_path is None:
        output_path = os.path.join(dataset_dir, "results_all.csv")
    output_fields = perf_fields + prof_fields if perf_rows else prof_fields
    _write_csv(output_path, output_fields, filtered_rows)

    print(f"Performance files: {len(perf_paths)} | Profile files: {len(prof_paths)}")
    print(f"Aggregated {len(merged_rows)} rows; kept {len(filtered_rows)} rows.")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
