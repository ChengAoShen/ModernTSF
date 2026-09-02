"""Lightweight, file-based research round state for Agent and human workflows."""

from __future__ import annotations

import json
import os
import re
import secrets
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from tsf_core.paths import working_root


ROUND_ENV = "MODERNTSF_RESEARCH_ROUND"
ROUND_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")
STATUSES = {"running", "blocked", "completed", "stopped"}
EVENT_KINDS = {
    "hypothesis",
    "decision",
    "observation",
    "run",
    "failure",
    "conclusion",
    "note",
}


class ResearchRoundError(ValueError):
    """Raised when a research round request is invalid or exceeds its budget."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def research_root() -> Path:
    """Return the research workspace without creating it."""
    configured = os.environ.get("TSF_WORK_DIR")
    work_dir = Path(configured) if configured else working_root() / "work_dirs"
    return work_dir.expanduser().resolve() / "_research"


def _round_dir(round_id: str) -> Path:
    if not ROUND_ID.fullmatch(round_id):
        raise ResearchRoundError("round ids must use lowercase letters, digits, and hyphens")
    return research_root() / round_id


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ResearchRoundError(f"unknown research round: {path.parent.name}") from exc


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


@contextmanager
def _state_lock(round_id: str, timeout: float = 10.0) -> Iterator[None]:
    lock = _round_dir(round_id) / ".lock"
    deadline = time.monotonic() + timeout
    while True:
        try:
            descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(descriptor)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise ResearchRoundError(f"research round {round_id!r} is busy")
            time.sleep(0.05)
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


def create_round(
    *,
    task: str,
    goal: str,
    max_runs: int | None = None,
    round_id: str | None = None,
) -> dict:
    """Create one research round with a small mutable state record."""
    if not task.strip() or not goal.strip():
        raise ResearchRoundError("task and goal must be non-empty")
    if max_runs is not None and max_runs < 1:
        raise ResearchRoundError("max_runs must be at least 1")
    candidate = round_id or (
        f"{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"
    )
    directory = _round_dir(candidate)
    try:
        directory.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ResearchRoundError(f"research round already exists: {candidate}") from exc
    now = _now()
    state = {
        "id": candidate,
        "task": task.strip(),
        "goal": goal.strip(),
        "status": "running",
        "max_runs": max_runs,
        "runs_used": 0,
        "created_at": now,
        "updated_at": now,
    }
    _write_json(directory / "round.json", state)
    (directory / "events.jsonl").touch()
    (directory / "logs").mkdir()
    add_event(candidate, "hypothesis", goal.strip())
    return state


def load_round(round_id: str) -> dict:
    """Load one round state record."""
    return _read_json(_round_dir(round_id) / "round.json")


def list_rounds() -> list[dict]:
    """List round state records, newest first."""
    records = []
    root = research_root()
    if not root.is_dir():
        return records
    for path in root.glob("*/round.json"):
        try:
            records.append(_read_json(path))
        except (ResearchRoundError, json.JSONDecodeError):
            continue
    return sorted(records, key=lambda item: item.get("created_at", ""), reverse=True)


def add_event(
    round_id: str,
    kind: str,
    text: str,
    *,
    details: dict | None = None,
) -> dict:
    """Append one concise research event."""
    if kind not in EVENT_KINDS:
        raise ResearchRoundError(f"unknown event kind {kind!r}")
    if not text.strip():
        raise ResearchRoundError("event text must be non-empty")
    load_round(round_id)
    event = {"time": _now(), "kind": kind, "text": text.strip()}
    if details:
        event["details"] = details
    path = _round_dir(round_id) / "events.jsonl"
    line = json.dumps(event, ensure_ascii=False, default=str) + "\n"
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        os.write(descriptor, line.encode("utf-8"))
    finally:
        os.close(descriptor)
    return event


def read_events(round_id: str) -> list[dict]:
    """Read all well-formed events for one round."""
    path = _round_dir(round_id) / "events.jsonl"
    load_round(round_id)
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


def set_status(round_id: str, status: str, message: str | None = None) -> dict:
    """Update a round lifecycle state and optionally append a conclusion."""
    if status not in STATUSES:
        raise ResearchRoundError(f"status must be one of {sorted(STATUSES)}")
    load_round(round_id)
    with _state_lock(round_id):
        state = load_round(round_id)
        state["status"] = status
        state["updated_at"] = _now()
        _write_json(_round_dir(round_id) / "round.json", state)
    if message:
        add_event(
            round_id,
            "conclusion" if status == "completed" else "note",
            message,
            details={"status": status},
        )
    return state


def claim_run(round_id: str, details: dict) -> int:
    """Atomically reserve one run from a round's optional run budget."""
    load_round(round_id)
    with _state_lock(round_id):
        state = load_round(round_id)
        if state["status"] != "running":
            raise ResearchRoundError(
                f"research round {round_id!r} is {state['status']}, not running"
            )
        used = int(state.get("runs_used", 0))
        limit = state.get("max_runs")
        if limit is not None and used >= int(limit):
            raise ResearchRoundError(
                f"research round {round_id!r} exhausted its {limit}-run budget"
            )
        number = used + 1
        state["runs_used"] = number
        state["updated_at"] = _now()
        _write_json(_round_dir(round_id) / "round.json", state)
    add_event(round_id, "run", f"Reserved run {number}", details=details)
    return number


def finish_run(
    round_id: str,
    number: int,
    *,
    status: str,
    run_id: str | None = None,
    metrics: dict | None = None,
    error: str | None = None,
) -> None:
    """Record the outcome of one claimed run without changing round status."""
    details = {"number": number, "status": status}
    if run_id:
        details["run_id"] = run_id
    if metrics:
        details["metrics"] = metrics
    if error:
        details["error"] = error
    kind = "run" if status == "passed" else "failure"
    text = f"Run {number} {status}" + (f": {run_id}" if run_id else "")
    add_event(round_id, kind, text, details=details)


def write_prompt(round_id: str, text: str) -> Path:
    """Store the rendered task prompt beside a round for direct Agent pickup."""
    load_round(round_id)
    path = _round_dir(round_id) / "prompt.txt"
    path.write_text(text, encoding="utf-8")
    return path


def write_log(round_id: str, name: str, text: str) -> Path:
    """Persist complete command output under a traversal-safe log name."""
    load_round(round_id)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("._") or "run"
    path = _round_dir(round_id) / "logs" / f"{safe_name}.log"
    if path.exists():
        path = path.with_name(f"{safe_name}-{secrets.token_hex(2)}.log")
    path.write_text(text, encoding="utf-8")
    return path


def events_for_run(run_id: str) -> list[dict]:
    """Return complete rounds containing one completed run."""
    matches = []
    for state in list_rounds():
        events = read_events(state["id"])
        if any(
            (event.get("details") or {}).get("run_id") == run_id
            for event in events
        ):
            matches.extend({"round": state["id"], **event} for event in events)
    return matches
