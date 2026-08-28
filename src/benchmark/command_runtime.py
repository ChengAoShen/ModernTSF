"""Shared process, model-slug, and trajectory helpers for public CLI commands."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

from tsf_core.paths import repository_root, working_root


ROOT = repository_root()
WORK_ROOT = working_root()
RUN_CONFIG_DIR = ROOT / "configs" / "runs"


def module_slug(name: str) -> str:
    """Normalize a public model name to the repository's lowercase module slug."""
    return re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")


def module_for_model(name: str) -> str:
    """Resolve a model name to its package module through the lazy catalog."""
    try:
        from benchmark.registry.models import MODEL_CATALOG

        path = MODEL_CATALOG.refs().get(name)
        if path:
            return path.split(".")[1]
    except Exception:
        pass
    return module_slug(name)


def trajectory():
    """Load the optional stdlib-only trajectory recorder, if available."""
    try:
        import benchmark.trajectory as recorder

        return recorder
    except Exception:
        return None


def passthrough(script: str, rest: list[str]) -> int:
    """Run one internal command module behind the stable public CLI surface."""
    module = script.removesuffix(".py")
    argv = [sys.executable, "-m", f"benchmark.commands.{module}", *rest]
    recorder = trajectory()
    if recorder is not None and recorder.is_active():
        return recorder.traced_run(argv, cwd=str(WORK_ROOT), label=f"command:{module}")
    return subprocess.run(argv, cwd=WORK_ROOT).returncode


def run_config(cfg: str, env_extra: dict | None = None) -> tuple[str, int, str]:
    """Execute one run configuration and return its path, status, and output tail."""
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    start_ts = time.time()
    config_path = Path(cfg)
    if not config_path.is_absolute() and not (WORK_ROOT / config_path).exists():
        packaged_candidate = ROOT / config_path
        if packaged_candidate.exists():
            config_path = packaged_candidate
    argv = [sys.executable, "-m", "benchmark.run_config", "--config", str(config_path)]
    proc = subprocess.run(
        argv,
        cwd=WORK_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    tail = next((line.strip() for line in reversed(output.splitlines()) if line.strip()), "")
    recorder = trajectory()
    if recorder is not None and recorder.is_active():
        recorder.record_command_result(
            argv=argv,
            cwd=str(WORK_ROOT),
            label="run",
            config_path=cfg,
            exit_code=proc.returncode,
            start_ts=start_ts,
            end_ts=time.time(),
            stdout=output,
        )
    return cfg, proc.returncode, tail
