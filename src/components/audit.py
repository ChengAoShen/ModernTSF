"""Static audit for the flat shared-component layer."""

from __future__ import annotations

import ast
from pathlib import Path

from components.catalog import COMPONENT_CATALOG


ROOT = Path(__file__).resolve().parents[2]
COMPONENTS = ROOT / "src" / "components"


def _component_imports(path: Path) -> set[str]:
    imports: set[str] = set()
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return imports
    for node in ast.walk(tree):
        module = None
        if isinstance(node, ast.ImportFrom):
            module = node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("components."):
                    imports.add(alias.name.split(".", 1)[1].split(".", 1)[0])
        if module and module.startswith("components."):
            imports.add(module.split(".", 1)[1].split(".", 1)[0])
    return imports


def components_used_by(package: Path) -> tuple[str, ...]:
    """Return shared component modules imported anywhere below a package."""
    used: set[str] = set()
    for path in package.rglob("*.py"):
        if path.name == "spec.py":
            continue
        used.update(_component_imports(path))
    return tuple(sorted(used - {"audit", "catalog"}))


def audit_components() -> list[str]:
    """Return catalog, filesystem, and consumer-import inconsistencies."""
    errors: list[str] = []
    catalog_names = set(COMPONENT_CATALOG.names())
    ignored = {"__init__", "audit", "catalog"}
    module_names = {path.stem for path in COMPONENTS.glob("*.py")} - ignored

    for name in sorted(catalog_names - module_names):
        errors.append(f"component catalog entry {name!r} has no module")
    for name in sorted(module_names - catalog_names):
        errors.append(f"components/{name}.py has no ComponentSpec")
    for spec in COMPONENT_CATALOG.specs():
        if spec.module != f"components.{spec.name}":
            errors.append(f"component {spec.name!r} has inconsistent module {spec.module!r}")
        if not spec.contract.strip():
            errors.append(f"component {spec.name!r} has no semantic contract")
        if not spec.keywords:
            errors.append(f"component {spec.name!r} has no retrieval keywords")
        if len(set(spec.keywords)) != len(spec.keywords):
            errors.append(f"component {spec.name!r} has duplicate retrieval keywords")

    consumers = list((ROOT / "src" / "models").rglob("*.py"))
    consumers += list(COMPONENTS.glob("*.py"))
    used: set[str] = set()
    for path in consumers:
        for name in _component_imports(path):
            if name in ignored:
                continue
            used.add(name)
            if name not in catalog_names:
                errors.append(
                    f"{path.relative_to(ROOT)} imports uncataloged component {name!r}"
                )

    for name in sorted(catalog_names - used - {"adj_norm"}):
        errors.append(f"component {name!r} has no model or component consumer")
    return errors


def main() -> int:
    errors = audit_components()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"Shared components OK: {len(COMPONENT_CATALOG.names())} modules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
