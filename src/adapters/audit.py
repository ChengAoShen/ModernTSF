"""Static consistency checks for shared adaptation backends."""

from __future__ import annotations

from pathlib import Path

from adapters.catalog import ADAPTER_CATALOG


ROOT = Path(__file__).resolve().parents[2]


def audit_adapters() -> list[str]:
    errors: list[str] = []
    package = ROOT / "src" / "adapters"
    expected_modules = {spec.module for spec in ADAPTER_CATALOG.values()}
    for name, spec in sorted(ADAPTER_CATALOG.items()):
        if name != spec.name:
            errors.append(f"adapter key {name!r} disagrees with spec name {spec.name!r}")
        module_file = ROOT / "src" / Path(*spec.module.split(".")).with_suffix(".py")
        if not module_file.is_file():
            errors.append(f"adapter {name!r} is missing {module_file.relative_to(ROOT)}")
        if not spec.contract or not spec.limitation:
            errors.append(f"adapter {name!r} must declare contract and limitation")
    files = {
        f"adapters.{path.stem}"
        for path in package.glob("*.py")
        if path.stem not in {"__init__", "audit", "catalog"}
    }
    for module in sorted(files - expected_modules):
        errors.append(f"uncataloged adapter module {module!r}")
    return errors


def main() -> int:
    errors = audit_adapters()
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"Shared adapters OK: {len(ADAPTER_CATALOG)} backends")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
