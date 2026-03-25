"""Command-line entrypoint for running benchmarks.

This module wires together config loading, registry setup, and
the sweep runner.
"""

from __future__ import annotations

import argparse

from benchmark.config import load_config
from benchmark.registry.loader import register_from_config
from benchmark.runner import run_sweep


def main() -> None:
    """Parse arguments and run the configured sweep.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(description="ModernTSF runner")
    parser.add_argument("--config", required=True, type=str, help="Path to config TOML")
    args = parser.parse_args()

    configs = load_config(args.config)
    for loaded in configs:
        register_from_config(loaded.config)
    run_sweep(configs)


if __name__ == "__main__":
    main()
