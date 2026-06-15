#!/usr/bin/env python3
"""ModernTSF unified Agent scaffold — one entry point for every tool.

Run any of the project's tools through a single command, plus concurrent
``smoke`` verification and concurrent multi-config ``run`` (these replace the old
``scripts/*.sh`` glue). Pure standard library (argparse + concurrent.futures +
subprocess) — no extra dependencies.

Usage:
    uv run python tool/tsf.py <command> [args...]

Scaffold:
    new-model        Scaffold a new model package + config + smoke config
    new-dataset      Scaffold a new dataset (custom / presplit / single)

Verify & run (concurrent):
    smoke            Run smoke config(s) and report PASS/FAIL  [--all|--model|--config] [--jobs N]
    run              Run experiment config(s) concurrently     [configs...] [--jobs N] [--gpus 0,1]

Results & plots:
    report           Generate a Markdown report (leaderboard + bubble chart + table)
    aggregate-plot   Aggregate a dataset's results + bubble chart (one shot)
    aggregate        -> tool/aggregate_results.py
    rank             -> tool/rank_models.py
    plot             -> tool/plot_bubble.py
    characteristics  -> tool/dataset_characteristics.py
    visualize        -> tool/visual_data.py
    predictions      -> tool/visualize_predictions.py
    inspect          -> tool/inspect_config.py

Data prep:
    pre-process      -> tool/pre_process.py
    convert-traffic  -> tool/convert_traffic.py
    gift-download    -> tool/gift_eval_download.py

TSEval contract & submit:
    schema-export    Export TSF-Core models to JSON Schema  [--out-dir DIR] [--check]
    trace            Trajectory capture session  (start [--label L] | end | status)
    submit           Package a run into a Submission Report  [then PR it to Diaugeia/TSEval]
    leaderboard-build  Aggregate submissions into a ranked leaderboard.json

Run `uv run python tool/tsf.py <command> --help` for a command's own flags.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tool"
RUN_CONFIG_DIR = ROOT / "configs" / "runs"

# Pass-through commands: command name -> tool script forwarded verbatim.
PASSTHROUGH = {
    "new-model": "new_model.py",
    "new-dataset": "new_dataset.py",
    "report": "report.py",
    "aggregate": "aggregate_results.py",
    "rank": "rank_models.py",
    "plot": "plot_bubble.py",
    "characteristics": "dataset_characteristics.py",
    "visualize": "visual_data.py",
    "predictions": "visualize_predictions.py",
    "inspect": "inspect_config.py",
    "pre-process": "pre_process.py",
    "convert-traffic": "convert_traffic.py",
    "gift-download": "gift_eval_download.py",
    "submit": "submit.py",
    "leaderboard-build": "leaderboard_build.py",
}


def _snake(name: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")


def _module_for_model(name: str) -> str:
    """Resolve a model name to its package module via MODEL_NAME_MAP (the
    authoritative source), falling back to a lowercased name."""
    try:
        from benchmark.registry.models import MODEL_NAME_MAP
        path = MODEL_NAME_MAP.get(name)
        if path:  # "models.<module>.registry"
            return path.split(".")[1]
    except Exception:
        pass
    return _snake(name)


def _trajectory():
    """Lazily import the (stdlib-only, torch-free) trajectory recorder.

    Returns the module, or ``None`` if it can't be imported — tracing then
    silently disables and commands run exactly as before."""
    try:
        import benchmark.trajectory as traj

        return traj
    except Exception:
        return None


def _passthrough(script: str, rest: list[str]) -> int:
    argv = [sys.executable, str(TOOL / script), *rest]
    traj = _trajectory()
    if traj is not None and traj.is_active():
        return traj.traced_run(argv, cwd=str(ROOT), label=f"passthrough:{script}")
    return subprocess.run(argv, cwd=ROOT).returncode


def _run_config(cfg: str, env_extra: dict | None = None) -> tuple[str, int, str]:
    """Run one config via the CLI; return (cfg, returncode, tail-of-output)."""
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    import time as _time

    start_ts = _time.time()
    proc = subprocess.run(
        ["modern-tsf", "--config", cfg],
        cwd=ROOT, env=env, capture_output=True, text=True,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    tail = ""
    for line in reversed(out.splitlines()):
        if line.strip():
            tail = line.strip()
            break
    traj = _trajectory()
    if traj is not None and traj.is_active():
        traj.record_command_result(
            argv=["modern-tsf", "--config", cfg], cwd=str(ROOT),
            label="run", config_path=cfg, exit_code=proc.returncode,
            start_ts=start_ts, end_ts=_time.time(), stdout=out,
        )
    return cfg, proc.returncode, tail


# --------------------------------------------------------------------------- #
# smoke
# --------------------------------------------------------------------------- #
def cmd_smoke(rest: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="tsf smoke",
                                 description="Run smoke config(s) concurrently and report PASS/FAIL.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--all", action="store_true", help="Run every configs/runs/smoke_*.toml")
    g.add_argument("--model", help="Run configs/runs/smoke_<snake(model)>.toml")
    g.add_argument("--config", nargs="+", help="Explicit smoke config path(s)")
    ap.add_argument("--jobs", type=int, default=min(8, (os.cpu_count() or 2)),
                    help="Concurrent workers (default: min(8, cpu))")
    args = ap.parse_args(rest)

    if args.all:
        configs = sorted(str(p.relative_to(ROOT)) for p in RUN_CONFIG_DIR.glob("smoke_*.toml"))
    elif args.model:
        configs = [f"configs/runs/smoke_{_module_for_model(args.model)}.toml"]
    elif args.config:
        configs = args.config
    else:
        ap.error("one of --all / --model / --config is required")

    missing = [c for c in configs if not (ROOT / c).exists()]
    if missing:
        print("Missing smoke config(s):", file=sys.stderr)
        for c in missing:
            print(f"  {c}", file=sys.stderr)
        return 1

    print(f"Running {len(configs)} smoke config(s) with {args.jobs} worker(s)...\n")
    results: list[tuple[str, int, str, float]] = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {}
        starts = {}
        for c in configs:
            starts[c] = time.monotonic()
            futs[ex.submit(_run_config, c)] = c
        for fut in as_completed(futs):
            cfg, code, tail = fut.result()
            dur = time.monotonic() - starts[cfg]
            name = Path(cfg).stem
            status = "PASS" if code == 0 else "FAIL"
            extra = "" if code == 0 else f"  (exit {code}) {tail[:80]}"
            print(f"  {status}  {name:<28} {dur:5.1f}s{extra}")
            results.append((cfg, code, tail, dur))

    passed = sum(1 for _, code, _, _ in results if code == 0)
    print(f"\n{passed}/{len(results)} passed")
    return 0 if passed == len(results) else 1


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #
def cmd_run(rest: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="tsf run",
                                 description="Run one or more experiment configs concurrently.")
    ap.add_argument("configs", nargs="*", default=["configs/runs/run_single_data.toml"],
                    help="TOML config path(s) (default: configs/runs/run_single_data.toml)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="Concurrent workers (default: 1; raise for parallel runs)")
    ap.add_argument("--gpus", default=None,
                    help="Comma-separated GPU ids, round-robined across jobs (CUDA_VISIBLE_DEVICES)")
    args = ap.parse_args(rest)

    configs = args.configs or ["configs/runs/run_single_data.toml"]
    missing = [c for c in configs if not (ROOT / c).exists()]
    if missing:
        print("Missing config(s):", file=sys.stderr)
        for c in missing:
            print(f"  {c}", file=sys.stderr)
        return 1

    gpus = [g.strip() for g in args.gpus.split(",")] if args.gpus else []

    def _env_for(idx: int) -> dict | None:
        if not gpus:
            return None
        return {"CUDA_VISIBLE_DEVICES": gpus[idx % len(gpus)]}

    print(f"Running {len(configs)} config(s) with {args.jobs} worker(s)"
          + (f", GPUs={gpus}" if gpus else "") + "...\n")
    results: list[tuple[str, int]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {ex.submit(_run_config, c, _env_for(i)): c for i, c in enumerate(configs)}
        for fut in as_completed(futs):
            cfg, code, tail = fut.result()
            status = "OK  " if code == 0 else "FAIL"
            extra = "" if code == 0 else f"  (exit {code}) {tail[:100]}"
            print(f"  {status}  {cfg}{extra}")
            results.append((cfg, code))

    ok = sum(1 for _, code in results if code == 0)
    print(f"\n{ok}/{len(results)} succeeded")
    return 0 if ok == len(results) else 1


# --------------------------------------------------------------------------- #
# aggregate-plot  (replaces scripts/aggregate_and_plot.sh)
# --------------------------------------------------------------------------- #
def cmd_aggregate_plot(rest: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="tsf aggregate-plot",
                                 description="Aggregate a dataset's results then plot a bubble chart.")
    ap.add_argument("--dataset", default="ETTh1")
    ap.add_argument("--pred-len", default="96")
    ap.add_argument("--x", default="latency_avg_ms")
    ap.add_argument("--y", default="mse")
    ap.add_argument("--size", default="total_params")
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--out-svg", default=None)
    args = ap.parse_args(rest)

    out_csv = args.out_csv or f"work_dirs/{args.dataset}/results_all.csv"
    out_svg = args.out_svg or f"work_dirs/plots/bubble_{args.dataset}_pl{args.pred_len}.svg"

    rc = _passthrough("aggregate_results.py", [
        "--dataset", args.dataset,
        "--filter", f"pred_len={args.pred_len}",
        "--perf-fields", "model,seq_len,pred_len,mse,mae",
        "--prof-fields", "latency_avg_ms,throughput_samples_sec,total_params,peak_vram_mb",
        "--output", out_csv,
    ])
    if rc != 0:
        return rc
    return _passthrough("plot_bubble.py", [
        "--csv", out_csv, "--x", args.x, "--y", args.y, "--size", args.size,
        "--size-scale", "log", "--x-scale", "log", "--y-scale", "log",
        "--color-by", "model", "--label-by", "model", "--output", out_svg,
    ])


def cmd_schema_export(rest: list[str]) -> int:
    """Export the TSF-Core contract models to JSON Schema — the only artifact
    TSEval consumes. Delegates to tsf_core.export (pydantic-only, no torch)."""
    from tsf_core.export import main as schema_main
    return schema_main(rest)


def cmd_trace(rest: list[str]) -> int:
    """Start / end / inspect a trajectory capture session."""
    ap = argparse.ArgumentParser(
        prog="tsf trace", description="Manage trajectory capture sessions."
    )
    ap.add_argument("action", choices=["start", "end", "status"])
    ap.add_argument("--label", default=None, help="Optional label for a new session")
    args = ap.parse_args(rest)

    traj = _trajectory()
    if traj is None:
        print("trajectory module unavailable", file=sys.stderr)
        return 1
    if args.action == "start":
        if traj.is_active():
            print(f"a session is already active: {traj.active_session()}")
            return 0
        print(f"trajectory session started: {traj.start(args.label)}")
        return 0
    if args.action == "end":
        sid = traj.end()
        print(f"trajectory session ended: {sid}" if sid else "no active session")
        return 0
    import json as _json

    print(_json.dumps(traj.status(), indent=2))
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd in PASSTHROUGH:
        return _passthrough(PASSTHROUGH[cmd], rest)
    if cmd == "smoke":
        return cmd_smoke(rest)
    if cmd == "run":
        return cmd_run(rest)
    if cmd == "aggregate-plot":
        return cmd_aggregate_plot(rest)
    if cmd == "schema-export":
        return cmd_schema_export(rest)
    if cmd == "trace":
        return cmd_trace(rest)
    print(f"unknown command: {cmd!r}\n", file=sys.stderr)
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
