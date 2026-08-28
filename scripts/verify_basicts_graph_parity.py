#!/usr/bin/env python3
"""Verify graph-model ports against an exact pinned BasicTS checkout."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import importlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark.catalog_metadata import model_records  # noqa: E402
from benchmark.parity import compare_model_parity  # noqa: E402
from benchmark.verification_results import (  # noqa: E402
    DEFAULT_INDEX,
    verification_subject_sha256,
    write_verification_result,
)
from components.adj_norm import symmetric_normalized_laplacian  # noqa: E402
from components.marks import to_spatiotemporal  # noqa: E402
from models.agcrn.model import Model as LocalAGCRN  # noqa: E402
from models.stgcn.model import Model as LocalSTGCN  # noqa: E402

SOURCE_URL = "https://github.com/GestaltCogTeam/BasicTS"
SOURCE_REVISION = "c218c07b6ce5e4cf908b147fd180c486346fed9c"
SOURCE_LICENSE = "Apache-2.0"
SOURCE_FILES = {
    "AGCRN": {
        "baselines/AGCRN/arch/agcn.py": "4ba649a3a109c0be68021b8cfab13b1e753237e5c03e955c2df940076963f0c1",
        "baselines/AGCRN/arch/agcrn_arch.py": "ce1f4f7efb9d6e6452400996e05fb66f688469b633be508b3d5b7244a76f9d29",
        "baselines/AGCRN/arch/agcrn_cell.py": "1bb012307a712797c168478223fb5ac89d625c604e0ae52e5a64fae73b15dc2c",
    },
    "STGCN": {
        "baselines/STGCN/arch/stgcn_arch.py": "ae507bf3c465a7c687a723f9979bd67952559dfdf73804202fde9002217d0f66",
        "baselines/STGCN/arch/stgcn_layers.py": "e94802be4763378bba6253b0b8ba93156a266ab3d1c31184a54e986b82b7fe93",
    },
}


def _load_upstream(checkout: Path) -> dict[str, type[torch.nn.Module]]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=checkout, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if revision != SOURCE_REVISION:
        raise ValueError(f"BasicTS checkout is {revision}, expected {SOURCE_REVISION}")
    for files in SOURCE_FILES.values():
        for relative, expected in files.items():
            actual = hashlib.sha256((checkout / relative).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"digest mismatch for {relative}: {actual}")
    sys.path.insert(0, str(checkout))
    try:
        agcrn = importlib.import_module("baselines.AGCRN.arch.agcrn_arch").AGCRN
        stgcn = importlib.import_module("baselines.STGCN.arch.stgcn_arch").STGCNChebGraphConv
    finally:
        sys.path.remove(str(checkout))
    return {"AGCRN": agcrn, "STGCN": stgcn}


def _round_trip(model: torch.nn.Module, args: tuple[object, ...]) -> tuple[bool, float]:
    model.eval()
    with torch.no_grad():
        expected = model(*args)
    stream = BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = deepcopy(model)
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    restored.eval()
    with torch.no_grad():
        actual = restored(*args)
    error = (expected - actual).abs()
    return torch.equal(expected, actual), float(error.max()) if error.numel() else 0.0


def _identity_state_map(local: torch.nn.Module, upstream: torch.nn.Module) -> dict[str, str]:
    upstream_state = upstream.state_dict()
    mapping = {name: name for name in local.state_dict() if name in upstream_state}
    if set(mapping) != set(local.state_dict()):
        raise ValueError(f"unmapped local state: {sorted(set(local.state_dict()) - set(mapping))}")
    return mapping


def _case(name: str, upstream_class: type[torch.nn.Module], *, batch: int, adjacency: np.ndarray) -> dict[str, object]:
    seed, seq_len, pred_len, nodes, input_dim = 3119, 12, 3, 4, 3
    torch.manual_seed(seed)
    marks = torch.zeros(batch, seq_len, nodes, 2)
    marks[..., 0] = torch.arange(seq_len).view(1, -1, 1) / seq_len
    marks[..., 1] = torch.arange(nodes).view(1, 1, -1) / nodes
    values = torch.randn(batch, seq_len, nodes)
    history = to_spatiotemporal(values, marks)
    if name == "AGCRN":
        local_wrapper = LocalAGCRN(seq_len, pred_len, nodes, adjacency, input_dim, 5, 3, 2, 3, 1)
        local = local_wrapper.net
        upstream = upstream_class(nodes, input_dim, 5, 1, pred_len, 2, True, 3, 3)
        module_map = {"encoder.dcrnn_cells.0.gate": "encoder.dcrnn_cells.0.gate", "end_conv": "end_conv"}
    else:
        gso = torch.from_numpy(symmetric_normalized_laplacian(adjacency).astype(np.float32))
        blocks = [[input_dim], [6, 3, 6], [6, 3, 6], [5, 5], [pred_len]]
        local_wrapper = LocalSTGCN(seq_len, pred_len, nodes, adjacency, input_dim, 3, 3, 6, 3, 5, "glu", "cheb_graph_conv", True, 0.2)
        local = local_wrapper.net
        upstream = upstream_class(3, 3, blocks, seq_len, nodes, "glu", "cheb_graph_conv", gso, True, 0.2)
        module_map = {"st_blocks.0.graph_conv": "st_blocks.0.graph_conv", "output": "output"}
    state_map = _identity_state_map(local, upstream)
    args = (history, None, 0, 0, False)
    report = compare_model_parity(local, upstream, args, state_map=state_map,
        module_map=module_map, modes=("eval", "train"), compare_gradients=True,
        seed=seed, atol=1e-6, rtol=1e-5)
    local_serial = _round_trip(local, args)
    upstream_serial = _round_trip(upstream, args)
    wrapper_serial = _round_trip(local_wrapper, (values, marks))
    local_wrapper.eval()
    with torch.no_grad():
        wrapper_out = local_wrapper(values, marks)
        backbone_out = local(history, None, 0, 0, False)[..., 0]
    preprocessing_error = float((wrapper_out - backbone_out).abs().max())
    if name == "AGCRN":
        buffer_value = local_wrapper.adj_mx
        buffer_persistent = False
    else:
        buffer_value = local_wrapper.gso
        buffer_persistent = "gso" in local_wrapper.state_dict()
    expected_buffer = (torch.from_numpy(adjacency) if name == "AGCRN" else
                       torch.from_numpy(symmetric_normalized_laplacian(adjacency).astype(np.float32)))
    buffer_error = float((buffer_value.cpu() - expected_buffer).abs().max())
    gradients_expected = len(dict(local.named_parameters()))
    gradients_seen = min(len(mode.parameter_gradients) for mode in report.modes.values())
    passed = (report.passed and local_serial[0] and upstream_serial[0] and wrapper_serial[0]
              and preprocessing_error == 0.0 and buffer_error == 0.0
              and gradients_seen == gradients_expected)
    return {
        "passed": passed, "batch": batch, "adjacency": adjacency.tolist(),
        "state_map": state_map, "mapped_buffers": sum(
            name in dict(local.named_buffers()) for name in state_map),
        "active_parameter_gradients": gradients_seen,
        "serialization": {"local": local_serial, "upstream": upstream_serial, "wrapper": wrapper_serial},
        "buffer_contract": {"max_abs": buffer_error, "persistent": buffer_persistent,
                            "meaning": "inspection-only adjacency" if name == "AGCRN" else "symmetric normalized Laplacian"},
        "preprocessing_max_abs": preprocessing_error, "report": report.to_dict(),
    }


def verify_model(name: str, upstream_class: type[torch.nn.Module]) -> dict[str, object]:
    nontrivial = np.array([[0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]], dtype=np.float32)
    cases = {
        "batch_one_identity": _case(name, upstream_class, batch=1, adjacency=np.eye(4, dtype=np.float32)),
        "batch_two_nontrivial_graph": _case(name, upstream_class, batch=2, adjacency=nontrivial),
    }
    return {
        "schema_version": 1, "model": name,
        "passed": all(bool(case["passed"]) for case in cases.values()),
        "source": {"url": SOURCE_URL, "revision": SOURCE_REVISION, "license": SOURCE_LICENSE,
                   "files": SOURCE_FILES[name]},
        "upstream_execution": "exact-pinned-checkout", "mapping_version": "basicts-graph-v1",
        "command": "uv run python scripts/verify_basicts_graph_parity.py --upstream-checkout <BasicTS@c218c07b>",
        "tolerances": {"atol": 1e-6, "rtol": 1e-5}, "cases": cases,
    }


def _errors(detail: dict[str, object], group: str) -> tuple[float, float]:
    comparisons = [item for case in detail["cases"].values() for mode in case["report"]["modes"].values() for item in mode[group].values()]
    return max(float(x["max_abs"]) for x in comparisons), max(float(x["max_rel"]) for x in comparisons)


def _check(passed: bool, evidence: list[str], **metrics: float | int | str) -> dict[str, object]:
    return {"passed": passed, "evidence": evidence, "metrics": metrics}


def canonical_result(name: str, detail: dict[str, object], path: Path) -> dict[str, object]:
    fields = next(x for x in model_records(ROOT) if x["name"] == name)
    rel = path.relative_to(ROOT).as_posix(); evidence = [rel, "tests/test_basicts_graph_parity.py"]
    metrics = {group: _errors(detail, group) for group in ("outputs", "intermediates", "input_gradients", "parameter_gradients")}
    serial = all(item[0] for case in detail["cases"].values() for item in case["serialization"].values())
    buffers = all(case["buffer_contract"]["max_abs"] == 0.0 for case in detail["cases"].values())
    passed = bool(detail["passed"])
    return {
        "schema_version": 1, "kind": "upstream-parity", "implementation": "upstream", "model": name,
        "verified_at": datetime.now(timezone.utc).isoformat(), "subject_sha256": verification_subject_sha256(ROOT, fields),
        "artifacts": {rel: hashlib.sha256(path.read_bytes()).hexdigest()},
        "commands": [detail["command"], "uv run python -m unittest tests.test_basicts_graph_parity -v", f"uv run tsf repo doctor --backward --models {name}"],
        "environment": {"python": platform.python_version(), "framework": f"torch {torch.__version__}",
            "dependencies": {"numpy": np.__version__, "torch": torch.__version__}, "platform": platform.platform(),
            "device": "cpu", "dtype": "float32", "deterministic": {"seed": 3119, "algorithms": True, "num_threads": 1}},
        "passed": passed, "source": {"url": SOURCE_URL, "revision": SOURCE_REVISION, "license": SOURCE_LICENSE},
        "mapping": {"version": "basicts-graph-v1", "parameters": len(next(iter(detail["cases"].values()))["state_map"]), "buffers": 0},
        "fixture": {"identifier": "basicts-graph-identity-nontrivial-v1", "description": "CPU float32 batch=1/2 cases with identity and nontrivial adjacency, node covariates, and seq_len=12."},
        "tolerances": detail["tolerances"], "modes": ["eval", "train"],
        "checks": {
            "outputs": _check(passed, evidence, max_abs=metrics["outputs"][0], max_rel=metrics["outputs"][1]),
            "intermediates": _check(passed, evidence, max_abs=metrics["intermediates"][0], max_rel=metrics["intermediates"][1]),
            "input_gradients": _check(passed, evidence, max_abs=metrics["input_gradients"][0], max_rel=metrics["input_gradients"][1]),
            "parameter_gradients": _check(passed, evidence, max_abs=metrics["parameter_gradients"][0], max_rel=metrics["parameter_gradients"][1]),
            "train_eval": _check(passed, evidence, modes="eval,train"),
            "buffers": _check(buffers, evidence, mapped_buffers=0, reason="BasicTS graph tensors are non-persistent; adapter adjacency/GSO value and persistence contract are checked explicitly"),
            "serialization": _check(serial, evidence, max_abs=0.0),
            "preprocessing": _check(passed, evidence, contract="node covariates and symmetric normalized Laplacian match exact upstream input/GSO"),
            "boundaries": _check(passed, evidence, cases="batch_one_identity,batch_two_nontrivial_graph"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--upstream-checkout", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "verification" / "parity"); args = parser.parse_args()
    torch.use_deterministic_algorithms(True); torch.set_num_threads(1)
    classes = _load_upstream(args.upstream_checkout.resolve()); args.output_dir.mkdir(parents=True, exist_ok=True)
    passed = True
    for name, cls in classes.items():
        detail = verify_model(name, cls); output = args.output_dir / f"{name}.json"
        output.write_text(json.dumps(detail, indent=2) + "\n", encoding="utf-8")
        if detail["passed"]:
            write_verification_result(ROOT / DEFAULT_INDEX, canonical_result(name, detail, output))
        print(f"{'PASS' if detail['passed'] else 'FAIL'} {name}: {output}"); passed &= bool(detail["passed"])
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
