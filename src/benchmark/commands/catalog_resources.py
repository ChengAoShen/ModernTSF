"""Public inspection commands for shared component and adapter catalogs."""

from __future__ import annotations

import json
import sys
from collections import Counter

from benchmark.command_runtime import ROOT, passthrough


def _print(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _model_evidence_record(fields: dict[str, object]) -> dict[str, object]:
    """Return one machine-readable evidence gate result."""
    paper = dict(fields.get("paper", {}))
    source = dict(fields.get("source", {}))
    evidence = str(fields.get("evidence", "unverified"))
    missing_source = [
        field
        for field in ("url", "revision", "license")
        if not source.get(field)
        or (field == "license" and source.get(field) == "NOASSERTION")
    ]
    blockers = []
    if not paper.get("title"):
        blockers.append("paper.title")
    if evidence == "unverified":
        blockers.append("verified evidence")
    if evidence == "upstream-port":
        blockers.extend(f"source.{field}" for field in missing_source)
    deviations = list(fields.get("deviations", ()))
    # A deliberately unverified port is still reviewed when its unresolved
    # differences are recorded.  This keeps review coverage distinct from the
    # stricter redistribution/reproduction evidence gate.
    reviewed = evidence != "unverified" or bool(deviations)
    return {
        "name": str(fields["name"]),
        "evidence": evidence,
        "reviewed": reviewed,
        "complete": not blockers,
        "blockers": blockers,
        "paper": {
            "title": paper.get("title", ""),
            "venue": paper.get("venue", ""),
            "year": paper.get("year"),
            "url": paper.get("url", ""),
        },
        "source": {
            "url": source.get("url", ""),
            "revision": source.get("revision", ""),
            "license": source.get("license", ""),
            "missing": missing_source,
        },
        "smoke_config": fields.get("smoke_config"),
        "components": list(fields.get("components", ())),
        "deviations": deviations,
    }


def model_command(args: list[str]) -> int:
    """Add, list, describe, or audit named model and method specifications."""
    if not args or args[0] in {"-h", "--help", "help"}:
        print(
            "usage: tsf model {add,list,show,audit} [args...]\n"
            "       tsf model list [--details | --json]"
        )
        return 0
    action, rest = args[0], args[1:]
    if action == "add":
        return passthrough("new_model.py", rest)

    from benchmark.registry.models import MODEL_CATALOG

    if action == "list":
        if any(arg not in {"--details", "--json"} for arg in rest) or len(rest) > 1:
            print("usage: tsf model list [--details | --json]", file=sys.stderr)
            return 2
        if not rest:
            print("\n".join(MODEL_CATALOG.names()))
            return 0
        from benchmark.catalog_metadata import model_records
        from benchmark.descriptions import read_model_card_description

        records = []
        for fields in model_records(ROOT):
            model_card = str(fields["model_card"])
            records.append(
                {
                    "name": str(fields["name"]),
                    "summary": read_model_card_description(ROOT / model_card).summary,
                    "evidence": fields.get("evidence", "unverified"),
                    "capabilities": sorted(fields.get("capabilities", ())),
                    "adapter": fields.get("adapter"),
                }
            )
        if rest == ["--json"]:
            _print(records)
        else:
            for record in records:
                print(f"{record['name']} [{record['evidence']}]\n  {record['summary']}")
        return 0
    if action == "show":
        if len(rest) != 1:
            print("usage: tsf model show <name>", file=sys.stderr)
            return 2
        from benchmark.descriptions import read_model_card_description

        spec = MODEL_CATALOG.get(rest[0])
        _print(
            {
                "name": spec.name,
                "module": spec.module,
                "summary": read_model_card_description(ROOT / spec.model_card).summary,
                "parameters": spec.params_schema.model_json_schema(),
                "paper": {
                    "title": spec.paper.title,
                    "venue": spec.paper.venue,
                    "year": spec.paper.year,
                    "url": spec.paper.url,
                },
                "source": {
                    "url": spec.source.url,
                    "revision": spec.source.revision,
                    "license": spec.source.license,
                },
                "evidence": spec.evidence,
                "config": spec.config_path,
                "model_card": spec.model_card,
                "smoke_config": spec.smoke_config,
                "capabilities": sorted(spec.capabilities),
                "adapter": spec.adapter,
                "components": list(spec.components),
                "output_type": spec.output_type,
                "deviations": list(spec.deviations),
            }
        )
        return 0
    if action == "audit":
        import argparse
        from benchmark.catalog_metadata import model_records

        parser = argparse.ArgumentParser(
            prog="tsf model audit",
            description="Report paper, source, smoke, and evidence-gate coverage.",
        )
        parser.add_argument("names", nargs="*", help="model names; default: all")
        output = parser.add_mutually_exclusive_group()
        output.add_argument("--json", action="store_true", help="emit per-model JSON")
        output.add_argument("--summary", action="store_true", help="emit aggregate JSON")
        parsed = parser.parse_args(rest)
        declared = {str(fields["name"]): fields for fields in model_records(ROOT)}
        names = parsed.names or sorted(declared)
        unknown = [name for name in names if name not in declared]
        if unknown:
            print(f"Unknown model(s): {', '.join(unknown)}", file=sys.stderr)
            return 2
        records = [_model_evidence_record(declared[name]) for name in names]
        failures = [record for record in records if not record["complete"]]
        if parsed.summary:
            blockers = Counter(
                blocker for record in failures for blocker in record["blockers"]
            )
            _print(
                {
                    "models": len(records),
                    "reviewed": sum(bool(r["reviewed"]) for r in records),
                    "unreviewed": sum(not r["reviewed"] for r in records),
                    "complete": len(records) - len(failures),
                    "incomplete": len(failures),
                    "evidence": dict(sorted(Counter(r["evidence"] for r in records).items())),
                    "incomplete_by_evidence": dict(
                        sorted(Counter(r["evidence"] for r in failures).items())
                    ),
                    "blockers": dict(sorted(blockers.items())),
                    "complete_source": sum(not r["source"]["missing"] for r in records),
                    "with_smoke_config": sum(bool(r["smoke_config"]) for r in records),
                }
            )
        elif parsed.json:
            _print(records)
        else:
            for record in failures:
                print(f"FAIL {record['name']}: {', '.join(record['blockers'])}")
            print(
                f"{len(records) - len(failures)}/{len(records)} "
                "model evidence records complete"
            )
        return 1 if failures else 0
    print(f"unknown model action: {action!r}", file=sys.stderr)
    return 2


def component_command(args: list[str]) -> int:
    """List, match, or describe shared components and their consumers."""
    from components.audit import components_used_by
    from components.catalog import COMPONENT_CATALOG
    from pathlib import Path

    if not args or args[0] in {"-h", "--help", "help"}:
        print(
            "usage: tsf component {list,show,match} [args...]\n"
            "       tsf component match <requirements...> [--limit N] [--json]"
        )
        return 0
    action, rest = args[0], args[1:]
    if action == "list" and not rest:
        for spec in COMPONENT_CATALOG.specs():
            print(f"{spec.name}\t{spec.contract}")
        return 0
    if action == "show" and len(rest) == 1:
        try:
            spec = COMPONENT_CATALOG.get(rest[0])
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        root = Path(__file__).resolve().parents[3]
        consumers = []
        for package in sorted((root / "src" / "models").iterdir()):
            if package.is_dir() and spec.name in components_used_by(package):
                consumers.append(package.name)
        _print(
            {
                "name": spec.name,
                "module": spec.module,
                "contract": spec.contract,
                "public_symbols": list(spec.public_symbols),
                "keywords": list(spec.keywords),
                "consumers": consumers,
            }
        )
        return 0
    if action == "match":
        import argparse

        parser = argparse.ArgumentParser(
            prog="tsf component match",
            description="Rank lexical component candidates; semantic review is still required.",
        )
        parser.add_argument("requirements", nargs="+", help="operations or contract terms")
        parser.add_argument("--limit", type=int, default=5)
        parser.add_argument("--json", action="store_true")
        parsed = parser.parse_args(rest)
        if parsed.limit < 1:
            parser.error("--limit must be positive")
        matches = COMPONENT_CATALOG.match(" ".join(parsed.requirements), parsed.limit)
        records = [
            {
                "name": match.spec.name,
                "score": match.score,
                "matched_terms": list(match.matched_terms),
                "contract": match.spec.contract,
                "module": match.spec.module,
                "review_required": True,
            }
            for match in matches
        ]
        if parsed.json:
            _print(records)
        else:
            for record in records:
                terms = ", ".join(record["matched_terms"])
                print(
                    f"{record['name']}\t{record['score']}\t{terms}\n"
                    f"  {record['contract']}"
                )
            if records:
                print(
                    "Candidate retrieval only; inspect the component contract and "
                    "implementation before reuse."
                )
        return 0
    print("usage: tsf component {list,show,match} [args...]", file=sys.stderr)
    return 2


def adapter_command(args: list[str]) -> int:
    """List approximation adapters or describe one adapter and its consumers."""
    from adapters.catalog import ADAPTER_CATALOG
    from benchmark.catalog_metadata import model_records
    from pathlib import Path

    if not args or args[0] in {"-h", "--help", "help"}:
        print("usage: tsf adapter {list,show} [name]")
        return 0
    action, rest = args[0], args[1:]
    if action == "list" and not rest:
        for name in sorted(ADAPTER_CATALOG):
            print(f"{name}\t{ADAPTER_CATALOG[name].contract}")
        return 0
    if action == "show" and len(rest) == 1:
        spec = ADAPTER_CATALOG.get(rest[0])
        if spec is None:
            print(f"Unknown adapter {rest[0]!r}", file=sys.stderr)
            return 2
        root = Path(__file__).resolve().parents[3]
        consumers = [
            str(record["name"])
            for record in model_records(root)
            if record.get("adapter") == spec.name
        ]
        _print(
            {
                "name": spec.name,
                "module": spec.module,
                "contract": spec.contract,
                "limitation": spec.limitation,
                "consumers": consumers,
            }
        )
        return 0
    print("usage: tsf adapter {list,show} [name]", file=sys.stderr)
    return 2
