"""Dataset and result resource command routing behind the public CLI."""

from __future__ import annotations

import sys

from benchmark.command_runtime import passthrough


def dataset_command(args: list[str]) -> int:
    """Route dataset scaffolding, preparation, inspection, and plotting."""
    if not args or args[0] in {"-h", "--help", "help"}:
        print("usage: tsf dataset {add,prepare,inspect,plot,convert-traffic,gift-download} [args...]")
        return 0
    action, rest = args[0], args[1:]
    scripts = {
        "add": "new_dataset.py",
        "prepare": "pre_process.py",
        "inspect": "dataset_characteristics.py",
        "plot": "visual_data.py",
        "convert-traffic": "convert_traffic.py",
        "gift-download": "gift_eval_download.py",
    }
    script = scripts.get(action)
    if script is None:
        print(f"unknown dataset action: {action!r}", file=sys.stderr)
        return 2
    return passthrough(script, rest)


def result_command(args: list[str]) -> int:
    """Route result aggregation, ranking, plotting, reporting, and visualization."""
    if not args or args[0] in {"-h", "--help", "help"}:
        print("usage: tsf result {aggregate,rank,plot,report,predictions} [args...]")
        return 0
    action, rest = args[0], args[1:]
    scripts = {
        "aggregate": "aggregate_results.py",
        "rank": "rank_models.py",
        "plot": "plot_bubble.py",
        "report": "report.py",
        "predictions": "visualize_predictions.py",
    }
    script = scripts.get(action)
    if script is None:
        print(f"unknown result action: {action!r}", file=sys.stderr)
        return 2
    return passthrough(script, rest)
