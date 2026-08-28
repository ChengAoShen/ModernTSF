"""Public inspection commands for the flat model and component catalogs."""

from __future__ import annotations

import json
import sys
from collections import Counter

from benchmark.command_runtime import ROOT, passthrough


def _print(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _model_audit_record(
    fields: dict[str, object],
) -> dict[str, object]:
    """Return one machine-readable implementation metadata gate result."""
    paper = dict(fields.get("paper", {}))
    codebase = dict(fields.get("codebase", {}))
    implementation = str(fields.get("implementation", ""))
    missing_source = [
        field
        for field in ("url", "revision", "license")
        if not codebase.get(field)
        or (field == "license" and codebase.get(field) == "NOASSERTION")
    ]
    blockers = []
    if not paper.get("title"):
        blockers.append("paper.title")
    if implementation == "upstream":
        blockers.extend(f"codebase.{field}" for field in missing_source)
        if codebase.get("usage") != "ported":
            blockers.append("codebase.usage=ported")
    elif implementation == "rewrite":
        if codebase.get("usage") not in {"none", "reference-only"}:
            blockers.append("codebase.usage=reference-only")
    else:
        blockers.append("implementation")
    from benchmark.verification import evidence_state

    state = evidence_state(ROOT, str(fields["name"]), fields)
    verification_status: dict[str, object] = {
        "status": state.status,
        "current": state.current,
        "evidence": state.evidence,
    }
    if state.detail:
        verification_status["detail"] = state.detail
    if state.status != "passed" or not state.current:
        blockers.append("verification.failed")
    return {
        "name": str(fields["name"]),
        "implementation": implementation,
        "passed": not blockers,
        "blockers": blockers,
        "paper": {
            "title": paper.get("title", ""),
            "venue": paper.get("venue", ""),
            "year": paper.get("year"),
            "url": paper.get("url", ""),
        },
        "codebase": {
            "url": codebase.get("url", ""),
            "revision": codebase.get("revision", ""),
            "license": codebase.get("license", ""),
            "usage": codebase.get("usage", ""),
            "missing": missing_source,
        },
        "smoke_config": fields.get("smoke_config"),
        "components": list(fields.get("components", ())),
        "verification": verification_status,
    }


def model_command(args: list[str]) -> int:
    """Add, list, describe, or audit named model and method specifications."""
    if not args or args[0] in {"-h", "--help", "help"}:
        print(
            "usage: tsf model {add,list,show,search,audit} [args...]\n"
            "       tsf model list [--details | --json]"
        )
        return 0
    action, rest = args[0], args[1:]
    if action == "add":
        return passthrough("new_model.py", rest)

    if action == "list":
        if any(arg not in {"--details", "--json"} for arg in rest) or len(rest) > 1:
            print("usage: tsf model list [--details | --json]", file=sys.stderr)
            return 2
        from benchmark.catalog_metadata import model_records

        fields_by_name = model_records(ROOT)
        if not rest:
            print("\n".join(sorted(str(fields["name"]) for fields in fields_by_name)))
            return 0
        from benchmark.descriptions import read_model_card_description

        records = []
        for fields in fields_by_name:
            model_card = str(fields["model_card"])
            records.append(
                {
                    "name": str(fields["name"]),
                    "summary": read_model_card_description(ROOT / model_card).summary,
                    "implementation": fields["implementation"],
                    "capabilities": sorted(fields.get("capabilities", ())),
                }
            )
        if rest == ["--json"]:
            _print(records)
        else:
            for record in records:
                print(f"{record['name']} [{record['implementation']}]\n  {record['summary']}")
        return 0
    if action == "show":
        if len(rest) != 1:
            print("usage: tsf model show <name>", file=sys.stderr)
            return 2
        from benchmark.catalog_metadata import model_records
        from benchmark.registry.models import MODEL_CATALOG

        spec = MODEL_CATALOG.get(rest[0])
        fields = next(
            record for record in model_records(ROOT) if record["name"] == spec.name
        )
        paper = dict(fields["paper"])
        codebase = dict(fields["codebase"])
        fields["card_text"] = (ROOT / str(fields["model_card"])).read_text(
            encoding="utf-8"
        )
        audit = _model_audit_record(fields)
        _print(
            {
                "name": spec.name,
                "module": spec.module,
                "summary": fields["summary"],
                "parameters": spec.params_schema.model_json_schema(),
                "paper": {
                    "title": paper["title"],
                    "venue": paper["venue"],
                    "year": paper["year"],
                    "url": paper["url"],
                },
                "codebase": codebase,
                "implementation": fields["implementation"],
                "config": spec.config_path,
                "model_card": spec.model_card,
                "smoke_config": spec.smoke_config,
                "capabilities": sorted(spec.capabilities),
                "components": list(spec.components),
                "output_type": spec.output_type,
                "verification": audit["verification"],
                "blockers": audit["blockers"],
            }
        )
        return 0
    if action == "search":
        import argparse
        import re
        from benchmark.catalog_metadata import model_records

        parser = argparse.ArgumentParser(
            prog="tsf model search",
            description="Search canonical model-card metadata and text.",
        )
        parser.add_argument("query", nargs="+", help="terms describing a method")
        parser.add_argument("--limit", type=int, default=10)
        parser.add_argument("--json", action="store_true")
        parsed = parser.parse_args(rest)
        if parsed.limit < 1:
            parser.error("--limit must be positive")
        terms = set(re.findall(r"[a-z0-9]+", " ".join(parsed.query).casefold()))
        matches = []
        for fields in model_records(ROOT):
            paper = dict(fields["paper"])
            card = (ROOT / str(fields["model_card"])).read_text(encoding="utf-8")
            surfaces = {
                "name": str(fields["name"]).casefold(),
                "summary": str(fields["summary"]).casefold(),
                "paper": str(paper["title"]).casefold(),
                "card": card.casefold(),
            }
            matched = {
                term for term in terms if any(term in text for text in surfaces.values())
            }
            if not matched:
                continue
            score = len(matched) * 100 + sum(
                8
                if term in surfaces["name"]
                else 4
                if term in surfaces["summary"]
                else 3
                if term in surfaces["paper"]
                else 1
                for term in matched
            )
            matches.append(
                {
                    "name": fields["name"],
                    "implementation": fields["implementation"],
                    "summary": fields["summary"],
                    "score": score,
                    "matched_terms": sorted(matched),
                }
            )
        matches.sort(key=lambda item: (-int(item["score"]), str(item["name"])))
        matches = matches[: parsed.limit]
        if parsed.json:
            _print(matches)
        else:
            for match in matches:
                print(
                    f"{match['name']} [{match['implementation']}] "
                    f"score={match['score']}\n  {match['summary']}"
                )
        return 0
    if action == "audit":
        import argparse
        from benchmark.catalog_metadata import model_records

        parser = argparse.ArgumentParser(
            prog="tsf model audit",
            description="Audit model cards and executable verification evidence.",
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
        for name in names:
            card = ROOT / str(declared[name]["model_card"])
            declared[name]["card_text"] = card.read_text(encoding="utf-8")
        records = [
            _model_audit_record(declared[name])
            for name in names
        ]
        failures = [record for record in records if not record["passed"]]
        if parsed.summary:
            blockers = Counter(
                blocker for record in failures for blocker in record["blockers"]
            )
            _print(
                {
                    "models": len(records),
                    "passed": len(records) - len(failures),
                    "failed": len(failures),
                    "implementation": dict(sorted(Counter(r["implementation"] for r in records).items())),
                    "failed_by_implementation": dict(
                        sorted(Counter(r["implementation"] for r in failures).items())
                    ),
                    "blockers": dict(sorted(blockers.items())),
                    "verification": dict(
                        sorted(
                            Counter(
                                str(r["verification"]["status"])
                                for r in records
                            ).items()
                        )
                    ),
                    "complete_upstream_codebase": sum(
                        r["implementation"] == "upstream" and not r["codebase"]["missing"]
                        for r in records
                    ),
                    "with_smoke_config": sum(bool(r["smoke_config"]) for r in records),
                }
            )
        elif parsed.json:
            _print(records)
        else:
            for record in failures:
                print(f"FAIL {record['name']}: {', '.join(record['blockers'])}")
            print(
                f"{len(records) - len(failures)}/{len(records)} model audits passed"
            )
        return 1 if failures else 0
    print(f"unknown model action: {action!r}", file=sys.stderr)
    return 2

def component_command(args: list[str]) -> int:
    """List, match, or describe shared components and their consumers."""
    from benchmark.catalog.component_audit import components_used_by
    from benchmark.catalog.components import COMPONENT_CATALOG
    from pathlib import Path

    if not args or args[0] in {"-h", "--help", "help"}:
        print(
            "usage: tsf component {list,show,match,audit} [args...]\n"
            "       tsf component list [--json]\n"
            "       tsf component match <requirements...> [--limit N] [--json]"
        )
        return 0
    action, rest = args[0], args[1:]
    if action == "audit":
        if rest:
            print("tsf component audit takes no arguments", file=sys.stderr)
            return 2
        from benchmark.resource_cards import audit_resource_cards
        from benchmark.catalog.component_audit import audit_components
        from tsf_core.paths import repository_root

        root = repository_root()
        failures = audit_components()
        failures.extend(
            error for error in audit_resource_cards(root) if "models/_components" in error
        )
        for failure in failures:
            print(f"ERROR: {failure}")
        total = len(COMPONENT_CATALOG.names())
        print(f"Component catalog/cards: {'PASS' if not failures else 'FAIL'} ({total} components)")
        return 1 if failures else 0
    if action == "list" and (not rest or rest == ["--json"]):
        records = [
            {
                "name": spec.name,
                "module": spec.module,
                "summary": spec.contract,
                "card": f"src/models/_components/{spec.name}/README.md",
            }
            for spec in COMPONENT_CATALOG.specs()
        ]
        if rest == ["--json"]:
            _print(records)
        else:
            for record in records:
                print(f"{record['name']}\t{record['summary']}")
        return 0
    if action == "show" and len(rest) == 1:
        try:
            spec = COMPONENT_CATALOG.get(rest[0])
        except KeyError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        from tsf_core.paths import repository_root

        root = repository_root()
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
                "card": f"src/models/_components/{spec.name}/README.md",
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
    print("usage: tsf component {list,show,match,audit} [args...]", file=sys.stderr)
    return 2
