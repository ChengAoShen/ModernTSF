"""CLI for small, explicit research round workspaces."""

from __future__ import annotations

import argparse
import json
import sys

from benchmark.research_round import (
    EVENT_KINDS,
    STATUSES,
    ResearchRoundError,
    add_event,
    create_round,
    list_rounds,
    load_round,
    read_events,
    set_status,
)


def _print(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def research_command(rest: list[str]) -> int:
    """Create, inspect, and update a lightweight research round."""
    parser = argparse.ArgumentParser(prog="tsf research")
    subparsers = parser.add_subparsers(dest="action", required=True)

    start = subparsers.add_parser("start", help="Create a research round")
    start.add_argument("--task", default="research", help="Short workflow name")
    start.add_argument("--goal", required=True, help="Falsifiable goal or question")
    start.add_argument("--max-runs", type=int, default=None)
    start.add_argument("--id", dest="round_id", default=None)

    listing = subparsers.add_parser("list", help="List research rounds")
    listing.add_argument("--json", action="store_true")

    show = subparsers.add_parser("show", help="Show state and events")
    show.add_argument("round_id")

    note = subparsers.add_parser("note", help="Append a structured event")
    note.add_argument("round_id")
    note.add_argument("--kind", choices=sorted(EVENT_KINDS), default="note")
    note.add_argument("--text", required=True)

    status = subparsers.add_parser("status", help="Update round status")
    status.add_argument("round_id")
    status.add_argument("status", choices=sorted(STATUSES))
    status.add_argument("--message")

    args = parser.parse_args(rest)
    try:
        if args.action == "start":
            _print(
                create_round(
                    task=args.task,
                    goal=args.goal,
                    max_runs=args.max_runs,
                    round_id=args.round_id,
                )
            )
        elif args.action == "list":
            records = list_rounds()
            if args.json:
                _print(records)
            else:
                for record in records:
                    limit = record["max_runs"] if record["max_runs"] is not None else "unlimited"
                    print(
                        f"{record['id']}  {record['status']:<9} "
                        f"runs={record['runs_used']}/{limit}  {record['task']}"
                    )
        elif args.action == "show":
            _print({"round": load_round(args.round_id), "events": read_events(args.round_id)})
        elif args.action == "note":
            _print(add_event(args.round_id, args.kind, args.text))
        elif args.action == "status":
            _print(set_status(args.round_id, args.status, args.message))
        return 0
    except ResearchRoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
