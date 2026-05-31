#!/usr/bin/env python3
"""Scaffold a new ModernTSF dataset in one command.

Three patterns:

* ``custom``   — a plain flat-multivariate CSV (a ``date`` column + numeric
  channels). NO code, just a config that reuses the built-in ``custom`` loader.
* ``presplit`` — you already have ``train.csv`` / ``val.csv`` / ``test.csv`` in
  one folder. NO code, just a config (``name = "presplit"``).
* ``single``   — an unusual layout / synthetic generation needing a bespoke
  ``_read_data``. Generates the dataset class, schema, ``DATASET_NAME_MAP``
  entry, and a config.

Examples
--------
    uv run python tool/new_dataset.py --name my_csv --pattern custom \
        --root-path ./dataset/my_csv --data-path my_csv.csv --target OT

    uv run python tool/new_dataset.py --name my_split --pattern presplit \
        --root-path ./dataset/my_split --target OT

    uv run python tool/new_dataset.py --name my_special --pattern single \
        --root-path ./dataset/my_special --data-path my_special.csv --target OT
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DS_DIR = ROOT / "src" / "data" / "datasets"
SCHEMA_DIR = ROOT / "src" / "data" / "schemas" / "datasets"
NAME_MAP_FILE = ROOT / "src" / "benchmark" / "registry" / "datasets.py"
DS_CONFIG_DIR = ROOT / "configs" / "datasets"


def _config_custom(name, root_path, data_path, target) -> str:
    return (
        "[dataset]\n"
        'name = "custom"\n'
        f'root_path = "{root_path}"\n'
        f'data_path = "{data_path}"\n\n'
        "[dataset.params]\n"
        f'target = "{target}"\n'
        "scale = true\n"
        "split_ratio = [0.7, 0.1, 0.2]\n"
    )


def _config_presplit(name, root_path, target) -> str:
    return (
        "[dataset]\n"
        'name = "presplit"\n'
        f'root_path = "{root_path}"\n'
        'data_path = ""\n\n'
        "[dataset.params]\n"
        f'target = "{target}"\n'
        "scale = true\n"
    )


def _config_single(name, root_path, data_path, target) -> str:
    return (
        "[dataset]\n"
        f'name = "{name}"\n'
        f'root_path = "{root_path}"\n'
        f'data_path = "{data_path}"\n\n'
        "[dataset.params]\n"
        f'target = "{target}"\n'
        "scale = true\n"
        "split_ratio = [0.7, 0.1, 0.2]\n"
    )


def _schema_single(name) -> str:
    cls = "".join(p.capitalize() for p in name.split("_"))
    return (
        f'"""Parameter schema for the {name} dataset."""\n\n'
        "from pydantic import BaseModel, Field\n\n\n"
        "class DatasetParameterConfig(BaseModel):\n"
        f'    """Validated {name} dataset parameters."""\n\n'
        "    target: str\n"
        "    scale: bool = True\n"
        "    split_ratio: list[float] = Field(default_factory=lambda: [0.7, 0.1, 0.2])\n"
    )


def _dataset_single(name) -> str:
    cls = "Dataset_" + "".join(p.capitalize() for p in name.split("_"))
    return f'''"""{name} dataset implementation (SCAFFOLD).

Replace the body of ``_read_data`` with the real loading logic. It must return
``(series_data, time_stamp)`` as numpy arrays for the requested split.
"""
from __future__ import annotations

from typing import Tuple, cast

import numpy as np
import pandas as pd

from data.schemas.datasets.{name} import DatasetParameterConfig
from benchmark.registry import DATASET_REGISTRY
from data.datasets.base import ForecastingDataset


class {cls}(ForecastingDataset):
    """The {name} dataset."""

    def _read_data(
        self,
        flag: str,
        features: str,
        target: str,
        split_ratio: tuple[float, float, float],
        scale: bool,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # TODO: replace with the real loader. This template reads a CSV with a
        # `date` column + numeric channels and splits it by ratio.
        df_raw = pd.read_csv(self.file_path)
        cols = [c for c in df_raw.columns if c != "date"]
        if features == "S":
            cols = [target]
        df_data = df_raw[cols]

        num_samples = len(df_data)
        border1, border2 = self._get_borders(flag, num_samples, split_ratio)

        if scale:
            train_len = int(split_ratio[0] / sum(split_ratio) * num_samples)
            data = self._apply_scaling(df_data.to_numpy(), train_len)
        else:
            data = df_data.to_numpy()

        df_stamp = df_raw[["date"]] if "date" in df_raw.columns else df_raw.iloc[:, :0].copy()
        if "date" not in df_stamp.columns:
            df_stamp["date"] = 0
        time_stamp = np.asarray(self._build_time_stamp(df_stamp))

        series_data = np.asarray(data[border1:border2])
        time_stamp = time_stamp[border1:border2]
        return cast(np.ndarray, series_data), cast(np.ndarray, time_stamp)


def register() -> None:
    """Register the {name} dataset."""
    DATASET_REGISTRY.register("{name}", {cls}, DatasetParameterConfig)
'''


def _insert_name_map(name: str) -> str:
    text = NAME_MAP_FILE.read_text()
    if f'"{name}":' in text:
        return "already-present"
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("DATASET_NAME_MAP"))
    close = next(i for i in range(start, len(lines)) if lines[i].rstrip() == "}")
    entry = f'    # Scaffolded by tool/new_dataset.py\n    "{name}": "data.datasets.{name}",'
    lines.insert(close, entry)
    NAME_MAP_FILE.write_text("\n".join(lines) + "\n")
    return "inserted"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="Dataset key / file name (snake_case)")
    ap.add_argument("--pattern", required=True, choices=["custom", "presplit", "single"])
    ap.add_argument("--root-path", default=None, help="Dataset folder (e.g. ./dataset/<name>)")
    ap.add_argument("--data-path", default="", help="CSV file name inside root-path (custom/single)")
    ap.add_argument("--target", default="OT", help="Target column name (default: OT)")
    ap.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = ap.parse_args()

    name = args.name
    root_path = args.root_path or f"./dataset/{name}"
    cfg_path = DS_CONFIG_DIR / f"{name}.toml"

    targets: dict[Path, str] = {}
    if args.pattern == "custom":
        targets[cfg_path] = _config_custom(name, root_path, args.data_path or f"{name}.csv", args.target)
    elif args.pattern == "presplit":
        targets[cfg_path] = _config_presplit(name, root_path, args.target)
    else:  # single
        targets[DS_DIR / f"{name}.py"] = _dataset_single(name)
        targets[SCHEMA_DIR / f"{name}.py"] = _schema_single(name)
        targets[cfg_path] = _config_single(name, root_path, args.data_path or f"{name}.csv", args.target)

    existing = [p for p in targets if p.exists()]
    if existing and not args.force:
        print("Refusing to overwrite existing files (use --force):", file=sys.stderr)
        for p in existing:
            print(f"  {p.relative_to(ROOT)}", file=sys.stderr)
        raise SystemExit(1)

    for path, content in targets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)

    status = "n/a"
    if args.pattern == "single":
        status = _insert_name_map(name)

    print(f"✓ Scaffolded dataset '{name}' (pattern: {args.pattern})")
    for path in targets:
        print(f"  + {path.relative_to(ROOT)}")
    if args.pattern == "single":
        print(f"  ~ DATASET_NAME_MAP: {status}")
    print()
    print("Next steps:")
    if args.pattern == "single":
        print(f"  1. Implement the loader in src/data/datasets/{name}.py (_read_data).")
    print(f"  - Put the data under {root_path}/ , then reference the config from a")
    print(f"    run config via `extends = [..., \"../datasets/{name}.toml\", ...]`.")


if __name__ == "__main__":
    main()
