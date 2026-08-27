"""Concurrent smoke-test and experiment execution commands."""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from benchmark.command_runtime import ROOT, RUN_CONFIG_DIR, module_for_model, run_config


def smoke_command(rest: list[str]) -> int:
    """Run selected end-to-end smoke configurations concurrently."""
    parser = argparse.ArgumentParser(
        prog="tsf smoke",
        description="Run smoke config(s) concurrently and report PASS/FAIL.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Run every smoke config")
    group.add_argument("--model", help="Run smoke configs belonging to one model")
    group.add_argument("--config", nargs="+", help="Explicit smoke config path(s)")
    parser.add_argument(
        "--jobs",
        type=int,
        default=min(8, os.cpu_count() or 2),
        help="Concurrent workers (default: min(8, cpu))",
    )
    args = parser.parse_args(rest)

    if args.all:
        configs = sorted(str(path.relative_to(ROOT)) for path in RUN_CONFIG_DIR.glob("smoke_*.toml"))
    elif args.model:
        prefix = module_for_model(args.model)
        try:
            from benchmark.registry.models import MODEL_CATALOG

            other_modules = {
                path.split(".")[1]
                for name, path in MODEL_CATALOG.refs().items()
                if module_for_model(name) != prefix
            }
        except Exception:
            other_modules = set()
        configs = sorted(
            str(path.relative_to(ROOT))
            for path in RUN_CONFIG_DIR.glob(f"smoke_{prefix}*.toml")
            if path.stem[len("smoke_") :] not in other_modules
        )
        if not configs:
            configs = [f"configs/runs/smoke_{prefix}.toml"]
    elif args.config:
        configs = args.config
    else:
        parser.error("one of --all / --model / --config is required")

    missing = [config for config in configs if not (ROOT / config).exists()]
    if missing:
        print("Missing smoke config(s):", file=sys.stderr)
        for config in missing:
            print(f"  {config}", file=sys.stderr)
        return 1

    print(f"Running {len(configs)} smoke config(s) with {args.jobs} worker(s)...\n")
    results: list[tuple[str, int, str, float]] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {}
        starts = {}
        for config in configs:
            starts[config] = time.monotonic()
            futures[executor.submit(run_config, config)] = config
        for future in as_completed(futures):
            config, code, tail = future.result()
            duration = time.monotonic() - starts[config]
            status = "PASS" if code == 0 else "FAIL"
            extra = "" if code == 0 else f"  (exit {code}) {tail[:80]}"
            print(f"  {status}  {Path(config).stem:<28} {duration:5.1f}s{extra}")
            results.append((config, code, tail, duration))

    passed = sum(1 for _, code, _, _ in results if code == 0)
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


def run_command(rest: list[str]) -> int:
    """Run one or more fully resolved experiment configurations."""
    parser = argparse.ArgumentParser(
        prog="tsf run",
        description="Run one or more experiment configs concurrently.",
    )
    parser.add_argument(
        "configs",
        nargs="*",
        default=["configs/runs/run_single_data.toml"],
        help="TOML config path(s)",
    )
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent workers")
    parser.add_argument("--gpus", default=None, help="Comma-separated GPU ids")
    args = parser.parse_args(rest)
    configs = args.configs or ["configs/runs/run_single_data.toml"]
    missing = [config for config in configs if not (ROOT / config).exists()]
    if missing:
        print("Missing config(s):", file=sys.stderr)
        for config in missing:
            print(f"  {config}", file=sys.stderr)
        return 1

    gpus = [gpu.strip() for gpu in args.gpus.split(",")] if args.gpus else []

    def environment(index: int) -> dict | None:
        return {"CUDA_VISIBLE_DEVICES": gpus[index % len(gpus)]} if gpus else None

    print(
        f"Running {len(configs)} config(s) with {args.jobs} worker(s)"
        + (f", GPUs={gpus}" if gpus else "")
        + "...\n"
    )
    results: list[tuple[str, int]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        futures = {
            executor.submit(run_config, config, environment(index)): config
            for index, config in enumerate(configs)
        }
        for future in as_completed(futures):
            config, code, tail = future.result()
            status = "OK  " if code == 0 else "FAIL"
            extra = "" if code == 0 else f"  (exit {code}) {tail[:100]}"
            print(f"  {status}  {config}{extra}")
            results.append((config, code))

    succeeded = sum(1 for _, code in results if code == 0)
    print(f"\n{succeeded}/{len(results)} succeeded")
    return 0 if succeeded == len(results) else 1
