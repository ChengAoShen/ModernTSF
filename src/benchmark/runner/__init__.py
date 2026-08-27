"""Single-run and sweep execution entry points."""

from benchmark.runner.run_one import run_one
from benchmark.runner.run_sweep import run_sweep

__all__ = ["run_one", "run_sweep"]
