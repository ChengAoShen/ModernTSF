#!/usr/bin/env python3
"""Scaffold a new ModernTSF model in one command.

Generates the full model package, a model config, and an end-to-end smoke run
config, and registers the model in ``MODEL_NAME_MAP`` — everything except the
actual architecture, which you fill into the generated ``model.py`` ``forward``.

Examples
--------
    # A plain (B, T, C) forecaster with two extra hyper-parameters
    uv run python tool/new_model.py --name MyModel \
        --params "enc_in:int,hidden:int=128,dropout:float=0.1"

    # A node-structured graph / spatiotemporal model (reads params["adj_mx"])
    uv run python tool/new_model.py --name MyGraphNet --graph \
        --params "enc_in:int,hidden:int=64"

After scaffolding:
    1. Fill the architecture into src/models/<module>/model.py (the `forward`).
    2. Verify end-to-end:  uv run python tool/tsf.py smoke --model <Name>
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "src" / "models"
NAME_MAP_FILE = ROOT / "src" / "benchmark" / "registry" / "models.py"
MODEL_CONFIG_DIR = ROOT / "configs" / "models"
RUN_CONFIG_DIR = ROOT / "configs" / "runs"

_PY_DEFAULTS = {"int": "0", "float": "0.0", "str": '""', "bool": "False"}


def _snake(name: str) -> str:
    """Derive the module name: lowercase, non-alphanumerics -> '_'.

    Matches the project convention (CARD -> card, AirFormer -> airformer,
    S_Mamba -> s_mamba). Pass --module to override.
    """
    return re.sub(r"[^0-9a-z]+", "_", name.lower()).strip("_")


def _parse_params(spec: str | None) -> list[tuple[str, str, str | None]]:
    """Parse 'enc_in:int,hidden:int=128' -> [(name, type, default_or_None)]."""
    out: list[tuple[str, str, str | None]] = []
    if not spec:
        return out
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        field, _, rest = chunk.partition(":")
        field = field.strip()
        typ, _, default = rest.partition("=")
        typ = (typ or "int").strip()
        default = default.strip() if "=" in rest else None
        if typ not in _PY_DEFAULTS:
            raise SystemExit(f"unsupported type {typ!r} for param {field!r} "
                             f"(use one of {sorted(_PY_DEFAULTS)})")
        out.append((field, typ, default))
    return out


def _toml_value(typ: str, default: str | None) -> str:
    if default is not None:
        return default if typ != "str" else f'"{default.strip(chr(34))}"'
    # representative placeholder for the config when no default was given
    return {"int": "128", "float": "0.1", "str": '"value"', "bool": "true"}[typ]


def _schema_py(name: str, params: list[tuple[str, str, str | None]], graph: bool) -> str:
    lines = [f'"""Parameter schema for the {name} model."""', "",
             "from pydantic import BaseModel", "", "",
             "class ModelParameterConfig(BaseModel):",
             f'    """Validated {name} parameters supplied via ``model.params``."""',
             "", "    enc_in: int"]
    if graph:
        lines.append("    cov_dim: int = 2")
    for field, typ, default in params:
        if field in ("enc_in", "cov_dim"):
            continue
        if default is None:
            lines.append(f"    {field}: {typ}")
        else:
            lit = default if typ != "str" else f'"{default.strip(chr(34))}"'
            lines.append(f"    {field}: {typ} = {lit}")
    return "\n".join(lines) + "\n"


def _registry_py(name: str, module: str, params, graph: bool) -> str:
    args = ["            seq_len=cfg.task.seq_len,",
            "            pred_len=cfg.task.pred_len,",
            '            enc_in=params["enc_in"],']
    if graph:
        args.append('            adj_mx=params.get("adj_mx"),')
        args.append('            cov_dim=params.get("cov_dim", 2),')
    for field, typ, default in params:
        if field in ("enc_in", "cov_dim"):
            continue
        if default is None:
            args.append(f'            {field}=params["{field}"],')
        else:
            lit = default if typ != "str" else f'"{default.strip(chr(34))}"'
            args.append(f'            {field}=params.get("{field}", {lit}),')
    body = "\n".join(args)
    return (
        f'"""Model registration for {name}."""\n\n'
        "from benchmark.registry import MODEL_REGISTRY\n"
        f"from models.{module}.model import Model\n"
        f"from models.{module}.schema import ModelParameterConfig\n\n\n"
        "def register() -> None:\n"
        f'    """Register the {name} model factory and parameter schema."""\n'
        "    MODEL_REGISTRY.register(\n"
        f'        "{name}",\n'
        "        lambda cfg, params: Model(\n"
        f"{body}\n"
        "        ),\n"
        "        ModelParameterConfig,\n"
        "    )\n"
    )


def _model_py(name: str, module: str, params, graph: bool) -> str:
    extra = "".join(
        f", {f}: {t} = {d if t != 'str' else f'{chr(34)}{d.strip(chr(34))}{chr(34)}'}"
        if d is not None else f", {f}: {t}"
        for f, t, d in params if f not in ("enc_in", "cov_dim")
    )
    if graph:
        return f'''"""ModernTSF adapter for the {name} spatiotemporal model (SCAFFOLD).

Consumes node-structured batches and returns ``(B, pred_len, N)``. The body
below is a trivial shape-correct PLACEHOLDER — replace it with the real
architecture. ``adj_mx`` (``(N, N)``) and the node covariates are available;
the placeholder ignores both.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from models._external.marks import to_spatiotemporal


class Model(nn.Module):
    """Adapter wrapping the {name} graph forecaster."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        adj_mx: "np.ndarray | None" = None,
        cov_dim: int = 2{extra},
    ) -> None:
        super().__init__()
        if adj_mx is None:
            adj_mx = np.eye(enc_in, dtype=np.float32)
        self.register_buffer("adj_mx", torch.as_tensor(adj_mx, dtype=torch.float32))
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.cov_dim = cov_dim
        # TODO: replace this placeholder with the real graph architecture.
        # The input tensor `st` is (B, T, N, 1 + cov_dim); use self.adj_mx for
        # message passing across the N nodes.
        self.placeholder = nn.Linear(seq_len, pred_len)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, *args):
        if x_mark_enc is None:
            x_mark_enc = x_enc.new_zeros((x_enc.shape[0], x_enc.shape[1], 6))
        st = to_spatiotemporal(x_enc, x_mark_enc)        # (B, T, N, 1 + F)
        value = st[..., 0]                               # (B, T, N)
        out = self.placeholder(value.permute(0, 2, 1))   # (B, N, pred_len)
        return out.transpose(1, 2)                       # (B, pred_len, N)
'''
    return f'''"""ModernTSF adapter for the {name} forecaster (SCAFFOLD).

Consumes ``(B, seq_len, enc_in)`` and returns ``(B, pred_len, enc_in)``. The
body below is a trivial shape-correct PLACEHOLDER (a per-channel linear map) —
replace it with the real architecture.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Model(nn.Module):
    """The {name} forecasting model."""

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int{extra},
    ) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        # TODO: replace this placeholder with the real architecture.
        self.placeholder = nn.Linear(seq_len, pred_len)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, *args):
        # x_enc: (B, seq_len, enc_in) -> (B, pred_len, enc_in)
        return self.placeholder(x_enc.transpose(1, 2)).transpose(1, 2)
'''


def _model_config(name: str, params, graph: bool) -> str:
    lines = ["[model]", f'name = "{name}"', "", "[model.params]"]
    lines.append(f"enc_in = {8 if graph else 7}")
    if graph:
        lines.append("cov_dim = 2")
    for field, typ, default in params:
        if field in ("enc_in", "cov_dim"):
            continue
        lines.append(f"{field} = {_toml_value(typ, default)}")
    return "\n".join(lines) + "\n"


def _smoke_config(name: str, module: str, graph: bool) -> str:
    if graph:
        return f'''# Scaffolded spatiotemporal smoke run for {name} on the CauAir/CCAQ node
# bundle (adj_mx + calendar covariates, CPU, 1 epoch).
extends = ["../base.toml", "../datasets/cauair_ccaq_st.toml", "../models/{name}.toml"]

[experiment]
description = "Smoke: spatiotemporal / {name}"

[experiment.runtime]
device = "cpu"
num_workers = 0

[task]
mode = "spatiotemporal"
seq_len = 24
label_len = 0
pred_len = 24
features = "M"

[training]
epochs = 1
batch_size = 8
loss = "mae"
patience = 1

[dataset]
name = "cauair_st"
root_path = "dataset/cauair_ccaq"
data_path = ""

[dataset.params]
input_dim = 3
npz_name = "his.npz"
scale = false
max_windows = 32

[model.params]
enc_in = 8
'''
    return f'''# Scaffolded smoke run for {name} on the tiny built-in smoke dataset
# (CPU, 1 epoch).
extends = ["../base.toml", "../datasets/smoke.toml", "../models/{name}.toml"]

[experiment]
description = "Smoke: {name}"

[experiment.runtime]
device = "cpu"
num_workers = 0

[task]
seq_len = 96
label_len = 0
pred_len = 12
features = "M"

[training]
epochs = 1
batch_size = 16
loss = "mae"
patience = 1

[model.params]
enc_in = 6
'''


def _insert_name_map(name: str, module: str) -> str:
    text = NAME_MAP_FILE.read_text()
    if f'"{name}":' in text:
        return "already-present"
    lines = text.splitlines()
    # find the closing brace of the MODEL_NAME_MAP dict (first line that is "}"
    # after the "MODEL_NAME_MAP = {" line)
    start = next(i for i, ln in enumerate(lines) if ln.startswith("MODEL_NAME_MAP"))
    close = next(i for i in range(start, len(lines)) if lines[i].rstrip() == "}")
    entry = f'    # Scaffolded by tool/new_model.py\n    "{name}": "models.{module}.registry",'
    lines.insert(close, entry)
    NAME_MAP_FILE.write_text("\n".join(lines) + "\n")
    return "inserted"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="Model name (PascalCase, e.g. MyModel)")
    ap.add_argument("--module", default=None, help="Package dir (snake_case); derived from --name if omitted")
    ap.add_argument("--params", default=None,
                    help='Comma-separated "field:type[=default]" list, e.g. "enc_in:int,hidden:int=128"')
    ap.add_argument("--graph", action="store_true",
                    help="Generate a node-structured graph / spatiotemporal model (reads params['adj_mx'])")
    ap.add_argument("--force", action="store_true", help="Overwrite existing files")
    args = ap.parse_args()

    name = args.name
    module = args.module or _snake(name)
    params = _parse_params(args.params)
    pkg = MODELS_DIR / module

    targets = {
        pkg / "model.py": _model_py(name, module, params, args.graph),
        pkg / "schema.py": _schema_py(name, params, args.graph),
        pkg / "registry.py": _registry_py(name, module, params, args.graph),
        MODEL_CONFIG_DIR / f"{name}.toml": _model_config(name, params, args.graph),
        RUN_CONFIG_DIR / f"smoke_{module}.toml": _smoke_config(name, module, args.graph),
    }

    existing = [p for p in targets if p.exists()]
    if existing and not args.force:
        print("Refusing to overwrite existing files (use --force):", file=sys.stderr)
        for p in existing:
            print(f"  {p.relative_to(ROOT)}", file=sys.stderr)
        raise SystemExit(1)

    pkg.mkdir(parents=True, exist_ok=True)
    for path, content in targets.items():
        path.write_text(content)

    status = _insert_name_map(name, module)

    print(f"✓ Scaffolded model '{name}' (module: {module}, graph={args.graph})")
    for path in targets:
        print(f"  + {path.relative_to(ROOT)}")
    print(f"  ~ MODEL_NAME_MAP: {status}")
    print()
    print("Next steps:")
    print(f"  1. Implement the architecture in src/models/{module}/model.py (forward).")
    print(f"  2. Verify end-to-end:  uv run python tool/tsf.py smoke --model {name}")


if __name__ == "__main__":
    main()
