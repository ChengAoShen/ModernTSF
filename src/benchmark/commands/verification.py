"""Unified executable verification entrypoint for local model implementations."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import sys

from tsf_core.paths import repository_root


def _strict_contract(name: str) -> dict[str, str] | None:
    from benchmark.model_contracts import audit_model_contracts

    failures = audit_model_contracts([name], strict=True)
    if not failures:
        return None
    failure = failures[0]
    return {"stage": failure.stage, "error": failure.error}


def _records(names: list[str], jobs: int, run_contracts: bool) -> list[dict[str, object]]:
    from benchmark.catalog_metadata import model_records
    from benchmark.independent_validation import evidence_state

    root = repository_root()
    fields = {str(record["name"]): record for record in model_records(root)}
    unknown = sorted(set(names) - fields.keys())
    if unknown:
        raise ValueError(f"unknown model(s): {', '.join(unknown)}")
    contract_failures: dict[str, dict[str, str] | None] = {}
    if run_contracts:
        if jobs == 1:
            contract_failures = {name: _strict_contract(name) for name in names}
        else:
            with ProcessPoolExecutor(max_workers=jobs) as executor:
                contract_failures = dict(zip(names, executor.map(_strict_contract, names)))
    records = []
    for name in names:
        state = evidence_state(root, name, fields[name])
        contract = contract_failures.get(name)
        status = "failed" if contract is not None else state.status
        records.append(
            {
                "model": name,
                "status": status,
                "evidence_status": state.status,
                "evidence": state.evidence,
                "detail": state.detail,
                "contract": "failed" if contract is not None else "passed" if run_contracts else "not-run",
                "contract_failure": contract,
            }
        )
    return records


def _emit(records: list[dict[str, object]], as_json: bool) -> int:
    if as_json:
        print(json.dumps(records, indent=2, sort_keys=True))
    else:
        for record in records:
            detail = f": {record['detail']}" if record["detail"] else ""
            print(f"{str(record['status']).upper()} {record['model']}{detail}")
        passed = sum(record["status"] == "passed" for record in records)
        print(f"Independent validation: {passed}/{len(records)} models passed")
    return 0 if all(record["status"] == "passed" for record in records) else 1


def verification_command(args: list[str]) -> int:
    """Run or inspect the final route-neutral model verification contract."""
    if not args or args[0] in {"-h", "--help", "help"}:
        print(
            "usage: tsf verify {model,stale,all,index} [args...]\n"
            "       tsf verify model <Name...> [--refresh] [--jobs N] [--no-runtime] [--json]\n"
            "       tsf verify stale [--json]\n"
            "       tsf verify all [--jobs N] [--no-runtime] [--json]\n"
            "       tsf verify index"
        )
        return 0
    action, rest = args[0], args[1:]
    if action == "index":
        if rest:
            print("tsf verify index takes no arguments", file=sys.stderr)
            return 2
        from benchmark.independent_validation import rebuild_index

        index = rebuild_index(repository_root())
        print(f"Indexed independent evidence for {len(index.models)} models")
        return 0
    if action not in {"model", "stale", "all"}:
        print("usage: tsf verify {model,stale,all,index} [args...]", file=sys.stderr)
        return 2

    parser = argparse.ArgumentParser(prog=f"tsf verify {action}")
    if action == "model":
        parser.add_argument("names", nargs="+")
        parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--no-runtime", action="store_true")
    parser.add_argument("--json", action="store_true")
    parsed = parser.parse_args(rest)
    if parsed.jobs < 1:
        parser.error("--jobs must be positive")

    from benchmark.registry.models import MODEL_CATALOG

    names = parsed.names if action == "model" else MODEL_CATALOG.names()
    if action == "model" and parsed.refresh:
        from benchmark.catalog_metadata import model_records
        from benchmark.independent_validation import refresh_evidence

        root = repository_root()
        fields = {str(record["name"]): record for record in model_records(root)}
        unknown = sorted(set(names) - fields.keys())
        if unknown:
            print(f"unknown model(s): {', '.join(unknown)}", file=sys.stderr)
            return 2
        try:
            for name in names:
                refresh_evidence(root, name, fields[name])
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    try:
        records = _records(names, parsed.jobs, not parsed.no_runtime and action != "stale")
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if action == "stale":
        records = [record for record in records if record["evidence_status"] != "passed"]
        if parsed.json:
            print(json.dumps(records, indent=2, sort_keys=True))
        else:
            for record in records:
                print(f"{str(record['evidence_status']).upper()} {record['model']}: {record['detail']}")
            print(f"Independent validation backlog: {len(records)} models")
        return 1 if records else 0
    return _emit(records, parsed.json)
