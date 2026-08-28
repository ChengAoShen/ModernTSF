#!/usr/bin/env python3
"""Public Agent CLI for ModernTSF.

The CLI is intentionally a thin router. Command behavior lives in focused
modules so the public surface stays stable while model, data, execution, and
repository concerns evolve independently.

Usage:
    tsf <command> [args...]

Catalog and resource operations:
    model            add, list, show, or audit a model specification
    component        list, match, or show a reusable implementation component
    dataset          add, prepare, inspect, or plot a dataset
    result           aggregate, rank, plot, or report results
    repo             audit or diagnose the repository
    agent            list, inspect, validate, or render bounded Agent tasks

Execution:
    smoke            run smoke configurations concurrently
    run              run experiment configurations concurrently
    inspect          preview resolved configuration expansion

Records and integration:
    trace            manage trajectory capture
    submit           package a run into a Submission Report
    schema-export    export TSF-Core JSON Schema
    leaderboard-build  recompute a leaderboard from submissions

Run ``tsf <command> --help`` for command-specific options.
"""

from __future__ import annotations

import sys

from benchmark.command_runtime import passthrough


def schema_export_command(rest: list[str]) -> int:
    """Export the lightweight TSF-Core contract models to JSON Schema."""
    from tsf_core.export import main as schema_main

    return schema_main(rest)


def main(argv: list[str] | None = None) -> int:
    """Dispatch one public command without importing unrelated heavy modules."""
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(__doc__)
        return 0

    command, rest = argv[0], argv[1:]
    if command in {"smoke", "run"}:
        from benchmark.commands.execution import run_command, smoke_command

        return smoke_command(rest) if command == "smoke" else run_command(rest)
    if command in {"model", "component"}:
        from benchmark.commands.catalog_resources import (
            component_command,
            model_command,
        )

        handlers = {
            "model": model_command,
            "component": component_command,
        }
        return handlers[command](rest)
    if command in {"dataset", "result"}:
        from benchmark.commands.data_results import dataset_command, result_command

        return dataset_command(rest) if command == "dataset" else result_command(rest)
    if command == "repo":
        from benchmark.commands.repository import repository_command

        return repository_command(rest)
    if command == "agent":
        from benchmark.commands.agent_tasks import agent_command

        return agent_command(rest)
    if command == "trace":
        from benchmark.commands.trajectory import trajectory_command

        return trajectory_command(rest)
    if command == "schema-export":
        return schema_export_command(rest)

    passthrough_commands = {
        "inspect": "inspect_config.py",
        "submit": "submit.py",
        "leaderboard-build": "leaderboard_build.py",
    }
    script = passthrough_commands.get(command)
    if script is not None:
        return passthrough(script, rest)

    print(f"unknown command: {command!r}\n", file=sys.stderr)
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
