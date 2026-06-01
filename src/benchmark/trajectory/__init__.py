"""Agent-agnostic trajectory capture at the ``tsf`` CLI boundary.

A *trajectory* is the append-only record of the experiment process — which
commands ran, against which configs, with what outcome — captured at the
subprocess boundary of the ``tsf`` tool. Because the capture point is the CLI
(not any agent's private protocol), the same trajectory is produced whether the
driver is Claude Code, Codex, OpenCode, or a human. The captured
``trajectory.jsonl`` is the audit evidence bundled into a Submission Report.
"""

from __future__ import annotations

from benchmark.trajectory.recorder import (
    active_session,
    end,
    is_active,
    record_command_result,
    record_event,
    start,
    status,
    traced_run,
)

__all__ = [
    "start",
    "end",
    "status",
    "is_active",
    "active_session",
    "record_event",
    "record_command_result",
    "traced_run",
]
