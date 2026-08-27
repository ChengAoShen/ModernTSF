"""Torch-free readers for runtime specs and canonical model-card metadata."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Literal, TypedDict, cast


Implementation = Literal["upstream", "rewrite"]


class PaperMetadata(TypedDict):
    title: str
    venue: str
    year: int | None
    url: str


class CodebaseMetadata(TypedDict):
    url: str
    revision: str
    license: str
    usage: str


def _literal(node: ast.expr):
    """Read literal values and frozensets used by runtime specifications."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "frozenset":
            return tuple(ast.literal_eval(node.args[0])) if node.args else ()
    return ast.literal_eval(node)


def declared_model_fields(spec_file: Path) -> dict[str, object]:
    """Read literal top-level ``ModelSpec`` runtime fields without imports."""
    tree = ast.parse(spec_file.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "SPEC" for target in node.targets):
            continue
        if not isinstance(node.value, ast.Call):
            return {}
        fields: dict[str, object] = {}
        for keyword in node.value.keywords:
            if keyword.arg is None:
                continue
            try:
                fields[keyword.arg] = _literal(keyword.value)
            except (ValueError, TypeError):
                continue
        return fields
    return {}


def _scalar(value: str) -> object:
    """Parse the deliberately small scalar subset used by model cards."""
    value = value.strip()
    if not value or value in {"null", "~"}:
        return None
    if value in {"true", "false"}:
        return value == "true"
    if value.startswith(('"', "'")):
        return json.loads(value) if value.startswith('"') else value[1:-1]
    try:
        return int(value)
    except ValueError:
        return value


def read_front_matter(path: Path) -> dict[str, object]:
    """Read a model card's nested YAML-like front matter without PyYAML."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError(f"{path} has no YAML front matter")
    try:
        end = lines.index("---", 1)
    except ValueError as exc:
        raise ValueError(f"{path} has unterminated YAML front matter") from exc
    result: dict[str, object] = {}
    section: dict[str, object] | None = None
    for line in lines[1:end]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if ":" not in line:
            raise ValueError(f"{path} has invalid front-matter line: {line!r}")
        key, raw = line.strip().split(":", 1)
        if indent == 0:
            if raw.strip():
                result[key] = _scalar(raw)
                section = None
            else:
                nested: dict[str, object] = {}
                result[key] = nested
                section = nested
        elif indent == 2 and section is not None:
            section[key] = _scalar(raw)
        else:
            raise ValueError(f"{path} has unsupported front-matter indentation")
    return result


def read_model_card(path: Path) -> dict[str, object]:
    """Validate and return the canonical metadata stored in one README."""
    fields = read_front_matter(path)
    required = {"name", "implementation", "summary", "paper", "codebase"}
    missing = sorted(required - fields.keys())
    if missing:
        raise ValueError(f"{path} missing front matter: {', '.join(missing)}")
    unexpected = sorted(fields.keys() - required)
    if unexpected:
        raise ValueError(f"{path} has unsupported front matter: {', '.join(unexpected)}")
    implementation = fields["implementation"]
    if implementation not in {"upstream", "rewrite"}:
        raise ValueError(f"{path} has invalid implementation={implementation!r}")
    for section_name, keys in {
        "paper": {"title", "venue", "year", "url"},
        "codebase": {"url", "revision", "license", "usage"},
    }.items():
        section = fields.get(section_name)
        if not isinstance(section, dict):
            raise ValueError(f"{path} front matter {section_name} must be a mapping")
        absent = sorted(keys - section.keys())
        if absent:
            raise ValueError(
                f"{path} missing {section_name} fields: {', '.join(absent)}"
            )
        extra = sorted(section.keys() - keys)
        if extra:
            raise ValueError(
                f"{path} has unsupported {section_name} fields: {', '.join(extra)}"
            )
    return fields


def model_records(root: Path) -> list[dict[str, object]]:
    """Merge canonical README metadata with non-descriptive runtime spec fields."""
    records: list[dict[str, object]] = []
    for path in (root / "src" / "models").glob("*/spec.py"):
        runtime = declared_model_fields(path)
        if not runtime:
            continue
        card_path = path.parent / "README.md"
        metadata = read_model_card(card_path)
        fields = {**runtime, **metadata}
        fields["package"] = path.parent.name
        fields["spec_file"] = str(path.relative_to(root))
        fields["model_card"] = str(card_path.relative_to(root))
        records.append(fields)
    return sorted(records, key=lambda record: str(record["name"]).casefold())


def paper_metadata(record: dict[str, object]) -> PaperMetadata:
    """Return the typed paper mapping from a model record."""
    return cast(PaperMetadata, record["paper"])


def codebase_metadata(record: dict[str, object]) -> CodebaseMetadata:
    """Return the typed codebase mapping from a model record."""
    return cast(CodebaseMetadata, record["codebase"])
