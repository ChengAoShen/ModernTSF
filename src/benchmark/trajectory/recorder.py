"""Session-scoped trajectory recorder (stdlib only).

A session is a directory ``work_dirs/_trajectory/<session_id>/`` holding an
append-only ``trajectory.jsonl`` plus ``meta.json``. The active session id is
pointed to by ``work_dirs/_trajectory/CURRENT``. While a session is active,
``traced_run`` captures each ``tsf`` subprocess invocation as one event; when
inactive, commands run untouched (live stdio, zero overhead).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone

SCHEMA_VERSION = "1.0"

# Guards appends to trajectory.jsonl so concurrent traced runs (tsf run --jobs N)
# don't interleave partial lines.
_WRITE_LOCK = threading.Lock()


def _work_dir() -> str:
    return os.environ.get("TSF_WORK_DIR", "work_dirs")


def _traj_root() -> str:
    return os.path.join(_work_dir(), "_trajectory")


def _current_pointer() -> str:
    return os.path.join(_traj_root(), "CURRENT")


def _session_dir(session_id: str) -> str:
    return os.path.join(_traj_root(), session_id)


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def active_session() -> str | None:
    """Return the active session id, or ``None`` if tracing is off."""
    pointer = _current_pointer()
    try:
        if os.path.exists(pointer):
            sid = open(pointer).read().strip()
            return sid or None
    except Exception:
        pass
    return None


def is_active() -> bool:
    return active_session() is not None


def start(label: str | None = None) -> str:
    """Begin a new trajectory session and mark it active. Returns its id."""
    sid = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}"
    sdir = _session_dir(sid)
    os.makedirs(sdir, exist_ok=True)
    meta = {
        "schema_version": SCHEMA_VERSION,
        "session_id": sid,
        "label": label,
        "started_at": _iso(time.time()),
        "cwd": os.getcwd(),
    }
    with open(os.path.join(sdir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    open(os.path.join(sdir, "trajectory.jsonl"), "a").close()
    os.makedirs(_traj_root(), exist_ok=True)
    with open(_current_pointer(), "w") as f:
        f.write(sid)
    return sid


def end() -> str | None:
    """End the active session (clear the pointer). Returns the ended id."""
    sid = active_session()
    if sid is None:
        return None
    try:
        os.remove(_current_pointer())
    except Exception:
        pass
    return sid


def status() -> dict:
    """Return a small status dict describing the active session (if any)."""
    sid = active_session()
    if sid is None:
        return {"active": False}
    jsonl = os.path.join(_session_dir(sid), "trajectory.jsonl")
    n = 0
    try:
        with open(jsonl) as f:
            n = sum(1 for line in f if line.strip())
    except Exception:
        pass
    return {"active": True, "session_id": sid, "n_events": n, "path": jsonl}


def record_event(event: dict, session_id: str | None = None) -> None:
    """Append one event to a session's ``trajectory.jsonl``. No-op if inactive."""
    sid = session_id or active_session()
    if sid is None:
        return
    try:
        jsonl = os.path.join(_session_dir(sid), "trajectory.jsonl")
        os.makedirs(os.path.dirname(jsonl), exist_ok=True)
        line = json.dumps(event, ensure_ascii=False, default=str) + "\n"
        with _WRITE_LOCK, open(jsonl, "a") as f:
            f.write(line)
    except Exception:
        pass


def _summarize(text: str, head: int = 40, tail: int = 40) -> str:
    lines = [ln for ln in (text or "").splitlines()]
    if len(lines) <= head + tail:
        return "\n".join(lines)
    return "\n".join(lines[:head] + [f"... ({len(lines) - head - tail} lines elided) ..."] + lines[-tail:])


def _run_ids_since(start_ts: float) -> list[str]:
    """Collect run_ids from record.json files written since ``start_ts``."""
    found: list[str] = []
    root = _work_dir()
    try:
        for dirpath, _dirs, files in os.walk(root):
            if os.path.basename(dirpath) != "records":
                continue
            for name in files:
                if not name.endswith(".json"):
                    continue
                fp = os.path.join(dirpath, name)
                try:
                    if os.path.getmtime(fp) + 1e-3 < start_ts:
                        continue
                    rid = json.load(open(fp)).get("run_id") or name[:-5]
                    if rid:
                        found.append(rid)
                except Exception:
                    continue
    except Exception:
        pass
    return sorted(set(found))


def record_command_result(
    *,
    argv: list[str],
    cwd: str,
    label: str,
    config_path: str | None,
    exit_code: int,
    start_ts: float,
    end_ts: float,
    stdout: str,
) -> None:
    """Record one normalized command event. No-op if tracing is inactive."""
    if not is_active():
        return
    event = {
        "schema_version": SCHEMA_VERSION,
        "event_id": uuid.uuid4().hex[:12],
        "session_id": active_session(),
        "seq": status().get("n_events", 0),
        "label": label,
        "command": " ".join(argv),
        "argv": argv,
        "cwd": cwd,
        "config_path": config_path,
        "ts_start": _iso(start_ts),
        "ts_end": _iso(end_ts),
        "duration_sec": round(end_ts - start_ts, 4),
        "exit_code": exit_code,
        "status": "ok" if exit_code == 0 else "error",
        "stdout_summary": _summarize(stdout),
        "run_ids": _run_ids_since(start_ts),
    }
    record_event(event)


def traced_run(
    argv: list[str],
    cwd: str | None = None,
    label: str = "",
    config_path: str | None = None,
    env: dict | None = None,
) -> int:
    """Run a subprocess, recording one trajectory event when a session is active.

    When inactive, the process inherits stdio (live streaming, no capture). When
    active, output is captured (so it can be summarized into the event) and then
    echoed back so the caller still sees it. Returns the process exit code.
    """
    if not is_active():
        return subprocess.run(argv, cwd=cwd, env=env).returncode

    start_ts = time.time()
    proc = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, text=True)
    end_ts = time.time()
    out = (proc.stdout or "") + (proc.stderr or "")
    # Echo captured output so traced runs look the same to the caller.
    sys.stdout.write(out)
    sys.stdout.flush()
    record_command_result(
        argv=argv,
        cwd=cwd or os.getcwd(),
        label=label,
        config_path=config_path,
        exit_code=proc.returncode,
        start_ts=start_ts,
        end_ts=end_ts,
        stdout=out,
    )
    return proc.returncode
