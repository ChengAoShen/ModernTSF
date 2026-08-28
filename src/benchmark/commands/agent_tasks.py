"""Public CLI for inspecting and rendering harness-neutral Agent tasks."""

from __future__ import annotations

import argparse
import json
import sys

from tsf_core.agent_tasks import (
    AgentTaskError,
    audit_tasks,
    list_tasks,
    load_task,
    render_task,
    render_text,
)


def _json(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def agent_command(args: list[str]) -> int:
    """Route ``tsf agent task`` commands without executing generated work."""
    if not args or args[0] in {"-h", "--help", "help"}:
        print("usage: tsf agent task {list,show,render,validate} [args...]")
        return 0
    if args[0] != "task":
        print("usage: tsf agent task {list,show,render,validate} [args...]", file=sys.stderr)
        return 2
    rest = args[1:]
    if not rest or rest[0] in {"-h", "--help", "help"}:
        print("usage: tsf agent task {list,show,render,validate} [args...]")
        return 0
    action, tail = rest[0], rest[1:]
    try:
        if action == "list":
            if tail not in ([], ["--json"]):
                raise AgentTaskError("usage: tsf agent task list [--json]")
            records = list_tasks()
            if tail == ["--json"]:
                _json(records)
            else:
                for record in records:
                    print(f"{record['name']}: {record['summary']}")
            return 0
        if action == "show":
            if len(tail) != 1:
                raise AgentTaskError("usage: tsf agent task show <name>")
            _json(load_task(tail[0]))
            return 0
        if action == "validate":
            if tail:
                for name in tail:
                    load_task(name)
                print(f"Agent tasks OK: {len(tail)} selected")
                return 0
            errors = audit_tasks()
            if errors:
                for error in errors:
                    print(f"ERROR: {error}")
                return 1
            print(f"Agent tasks OK: {len(list_tasks())} templates")
            return 0
        if action == "render":
            parser = argparse.ArgumentParser(prog="tsf agent task render")
            parser.add_argument("name")
            parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
            parser.add_argument("--json", action="store_true")
            parsed = parser.parse_args(tail)
            supplied: dict[str, str] = {}
            for item in parsed.set:
                if "=" not in item or not item.split("=", 1)[0]:
                    raise AgentTaskError("--set requires KEY=VALUE")
                key, value = item.split("=", 1)
                supplied[key] = value
            payload = render_task(parsed.name, supplied)
            _json(payload) if parsed.json else print(render_text(payload), end="")
            return 0
    except AgentTaskError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"unknown Agent task action: {action}", file=sys.stderr)
    return 2
