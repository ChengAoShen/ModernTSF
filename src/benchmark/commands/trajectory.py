"""Start, finish, or inspect replayable Agent trajectory capture sessions."""

from __future__ import annotations

import argparse
import json
import sys

from benchmark.command_runtime import trajectory


def trajectory_command(rest: list[str]) -> int:
    """Manage the optional trajectory recorder through the public CLI."""
    parser = argparse.ArgumentParser(
        prog="tsf trace",
        description="Manage trajectory capture sessions.",
    )
    parser.add_argument("action", choices=["start", "end", "status"])
    parser.add_argument("--label", default=None, help="Optional label for a new session")
    args = parser.parse_args(rest)

    recorder = trajectory()
    if recorder is None:
        print("trajectory module unavailable", file=sys.stderr)
        return 1
    if args.action == "start":
        if recorder.is_active():
            print(f"a session is already active: {recorder.active_session()}")
            return 0
        print(f"trajectory session started: {recorder.start(args.label)}")
        return 0
    if args.action == "end":
        session_id = recorder.end()
        print(
            f"trajectory session ended: {session_id}"
            if session_id
            else "no active session"
        )
        return 0
    print(json.dumps(recorder.status(), indent=2))
    return 0
