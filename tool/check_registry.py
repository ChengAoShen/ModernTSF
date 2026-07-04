"""Static consistency check between configs/models/*.toml and MODEL_NAME_MAP.

Catches the two ways a model can drift out of sync without any single step
failing loudly: a config file with no registry entry (dead config), or a
registry entry whose module/config doesn't exist (dead entry). Pure file/
import-spec checks — no torch import, no model construction — so it runs in
under a second and is safe to call from CI.

Dataset registration isn't checked here: DATASET_NAME_MAP is many-to-one
(several configs share a `name` like "custom" or "cauair_st"), so "one config
per registry key" doesn't apply the way it does for models.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_CONFIG_DIR = ROOT / "configs" / "models"
SRC = ROOT / "src"


def _module_file(module_path: str) -> Path:
    return SRC / Path(*module_path.split(".")).with_suffix(".py")


def check() -> list[str]:
    from benchmark.registry.models import MODEL_NAME_MAP

    problems = []

    config_names = {p.stem for p in MODEL_CONFIG_DIR.glob("*.toml")}
    registry_names = set(MODEL_NAME_MAP)

    for name in sorted(registry_names - config_names):
        problems.append(
            f"'{name}' is in MODEL_NAME_MAP but has no configs/models/{name}.toml"
        )
    for name in sorted(config_names - registry_names):
        problems.append(
            f"configs/models/{name}.toml exists but '{name}' is not in MODEL_NAME_MAP"
        )

    seen_modules: dict[str, str] = {}
    for name, module_path in sorted(MODEL_NAME_MAP.items()):
        if importlib.util.find_spec(module_path) is None:
            problems.append(
                f"'{name}' -> '{module_path}' does not resolve to a module "
                f"({_module_file(module_path)} missing)"
            )
        prior = seen_modules.get(module_path)
        if prior is not None:
            problems.append(
                f"'{name}' and '{prior}' both point at module '{module_path}'"
            )
        else:
            seen_modules[module_path] = name

    return problems


def main() -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    problems = check()
    if not problems:
        num_configs = len(list(MODEL_CONFIG_DIR.glob("*.toml")))
        print(f"OK: {num_configs} configs/models/*.toml all match MODEL_NAME_MAP 1:1.")
        return 0
    print(f"Found {len(problems)} registry inconsistency(ies):")
    for p in problems:
        print(f"  - {p}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
