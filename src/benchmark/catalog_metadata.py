"""Torch-free readers for static metadata declared by model specifications."""

from __future__ import annotations

import ast
from pathlib import Path


def _literal(node: ast.expr):
    """Read literal values and the small metadata constructors used by specs."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id == "frozenset":
            return tuple(ast.literal_eval(node.args[0])) if node.args else ()
        if node.func.id in {"PaperRef", "SourceRef"}:
            return {
                keyword.arg: _literal(keyword.value)
                for keyword in node.keywords
                if keyword.arg is not None
            }
    return ast.literal_eval(node)


def declared_model_fields(spec_file: Path) -> dict[str, object]:
    """Read literal top-level ``ModelSpec`` fields without importing the model."""
    tree = ast.parse(spec_file.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "SPEC"
            for target in node.targets
        ):
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


def model_records(root: Path) -> list[dict[str, object]]:
    """Return every flat model's literal metadata in public-name order."""
    records: list[dict[str, object]] = []
    for path in (root / "src" / "models").glob("*/spec.py"):
        fields = declared_model_fields(path)
        if fields:
            fields["package"] = path.parent.name
            fields["spec_file"] = str(path.relative_to(root))
            records.append(fields)
    return sorted(records, key=lambda record: str(record["name"]).casefold())
