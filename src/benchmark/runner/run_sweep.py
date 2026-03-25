"""Sweep runner that executes multiple expanded configs."""

from __future__ import annotations

from typing import Iterable

from benchmark.runner.run_one import run_one


def run_sweep(configs: Iterable) -> list:
    """Run a list of expanded configs and collect results.

    Parameters
    ----------
    configs : Iterable
        Iterable of LoadedConfig instances.

    Returns
    -------
    list
        List of RunResult objects from each run.
    """
    results = []
    for loaded in configs:
        results.append(
            run_one(loaded.config, loaded.raw, loaded.sweep_keys, loaded.config_name)
        )
    return results
