#!/usr/bin/env python3
"""Verify GWNet, STNorm, and STDN against a pinned BasicTS checkout."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import platform
import subprocess
import sys
import types

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
from models._components.graph_utils import adj_to_supports  # noqa: E402
from models._components.marks import to_spatiotemporal  # noqa: E402
from models.gwnet._upstream import GraphWaveNet as LocalGWNet  # noqa: E402
from models.gwnet.model import Model as GWNetWrapper  # noqa: E402
from models.stdn._upstream import STDN as LocalSTDN  # noqa: E402
from models.stdn.model import Model as STDNWrapper  # noqa: E402
from models.stnorm.model import Model as STNormWrapper, STNorm as LocalSTNorm  # noqa: E402


SOURCE_URL = "https://github.com/GestaltCogTeam/BasicTS"
SOURCE_REVISION = "c218c07b6ce5e4cf908b147fd180c486346fed9c"
SOURCE_LICENSE = "Apache-2.0"
LICENSE_SHA256 = "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4"
SOURCE_FILES = {
    "GWNet": {"baselines/GWNet/arch/gwnet_arch.py": "de17f1741fe4164331957e4ae29804b65ee55b81ce793ee99a84be55198f026c"},
    "STNorm": {"baselines/STNorm/arch/stnorm_arch.py": "895838c3536ee5328c03e8ba93c3c1312725c447380d4003ccaef5dd0aeed747"},
    "STDN": {"baselines/STDN/arch/model.py": "d94d32ba4e5e1c62a6779632180603da2c993802a147c7adea66a7718d91a615"},
}
SEED = 5279


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_file(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_upstream(checkout: Path) -> dict[str, type[torch.nn.Module]]:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=checkout, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if revision != SOURCE_REVISION:
        raise ValueError(f"BasicTS checkout is {revision}, expected {SOURCE_REVISION}")
    if _sha(checkout / "LICENSE") != LICENSE_SHA256:
        raise ValueError("BasicTS LICENSE digest mismatch")
    for files in SOURCE_FILES.values():
        for relative, expected in files.items():
            actual = _sha(checkout / relative)
            if actual != expected:
                raise ValueError(f"source digest mismatch for {relative}: {actual}")

    # STDN imports torch_geometric only for inactive helpers. Supplying import
    # placeholders lets the exact pinned file execute without adding a runtime
    # dependency; the active STDN graph path uses its in-file gcn class.
    placeholders = {
        key: sys.modules.get(key)
        for key in ("torch_geometric", "torch_geometric.nn", "torch_geometric.data")
    }
    package = types.ModuleType("torch_geometric")
    nn_module = types.ModuleType("torch_geometric.nn")
    data_module = types.ModuleType("torch_geometric.data")
    nn_module.GCNConv = object
    data_module.Data = object
    package.nn = nn_module
    package.data = data_module
    sys.modules.update({
        "torch_geometric": package,
        "torch_geometric.nn": nn_module,
        "torch_geometric.data": data_module,
    })
    try:
        gw = _load_file("moderntsf_exact_gwnet", checkout / next(iter(SOURCE_FILES["GWNet"]))).GraphWaveNet
        sn = _load_file("moderntsf_exact_stnorm", checkout / next(iter(SOURCE_FILES["STNorm"]))).STNorm
        sd = _load_file("moderntsf_exact_stdn", checkout / next(iter(SOURCE_FILES["STDN"]))).STDN
    finally:
        for key, value in placeholders.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    return {"GWNet": gw, "STNorm": sn, "STDN": sd}


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


def _state_map(local: torch.nn.Module, upstream: torch.nn.Module) -> tuple[dict[str, str], list[str]]:
    left, right = local.state_dict(), upstream.state_dict()
    mapping = {name: name for name in left if name in right}
    if set(mapping) != set(left):
        raise ValueError(f"unmapped local state: {sorted(set(left) - set(mapping))}")
    for name in mapping:
        if left[name].shape != right[name].shape:
            raise ValueError(f"state shape mismatch for {name}")
    return mapping, sorted(set(right) - set(mapping.values()))


def _activity(model: torch.nn.Module, args: tuple[object, ...], seed: int) -> list[str]:
    cloned = tuple(
        value.detach().clone().requires_grad_(value.is_floating_point())
        if torch.is_tensor(value) else value for value in args
    )
    model.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    model(*cloned).float().sum().backward()
    return sorted(name for name, parameter in model.named_parameters() if parameter.grad is not None)


def _activity_contract(local: torch.nn.Module, upstream: torch.nn.Module,
                       args: tuple[object, ...]) -> dict[str, object]:
    modes = {}
    for mode in ("eval", "train"):
        local.train(mode == "train")
        upstream.train(mode == "train")
        left, right = _activity(local, args, SEED), _activity(upstream, args, SEED)
        modes[mode] = {"local": left, "upstream": right, "matched": left == right}
    active = sorted(set().union(*(entry["local"] for entry in modes.values())))
    return {"modes": modes, "active": active,
            "inactive_upstream": sorted(set(dict(upstream.named_parameters())) - set(active))}


def _marks(batch: int, seq: int, nodes: int) -> torch.Tensor:
    marks = torch.zeros(batch, seq, nodes, 2)
    marks[..., 0] = (torch.arange(seq).view(1, -1, 1) % 24) / 24.0
    marks[..., 1] = (torch.arange(seq).view(1, -1, 1) % 7) / 7.0
    return marks


def _common_finish(local: torch.nn.Module, upstream: torch.nn.Module,
                   wrapper: torch.nn.Module, backbone_args: tuple[object, ...],
                   wrapper_args: tuple[object, ...], module_map: dict[str, str],
                   preprocessing_error: float) -> dict[str, object]:
    mapping, upstream_only = _state_map(local, upstream)
    report = compare_model_parity(
        local, upstream, backbone_args, state_map=mapping, module_map=module_map,
        seed=SEED, atol=1e-6, rtol=1e-5,
    )
    activity = _activity_contract(local, upstream, backbone_args)
    inactive_upstream = set(activity["inactive_upstream"])
    upstream_only_inactive = set(upstream_only) <= inactive_upstream
    serial = {
        "local": _round_trip(local, backbone_args),
        "upstream": _round_trip(upstream, backbone_args),
        "wrapper": _round_trip(wrapper, wrapper_args),
    }
    expected = len(activity["active"])
    seen = min(len(mode.parameter_gradients) for mode in report.modes.values())
    local_buffers, upstream_buffers = dict(local.named_buffers()), dict(upstream.named_buffers())
    buffer_errors = [
        float((local_buffers[name] - upstream_buffers[mapping[name]]).abs().max())
        for name in mapping if name in local_buffers
    ]
    buffer_max_abs = max(buffer_errors, default=0.0)
    passed = (
        report.passed
        and all(item[0] for item in serial.values())
        and preprocessing_error == 0.0
        and all(item["matched"] for item in activity["modes"].values())
        and upstream_only_inactive
        and seen == expected
        and buffer_max_abs == 0.0
    )
    mapped_parameters = sum(name in dict(local.named_parameters()) for name in mapping)
    return {"passed": passed, "state_map": mapping, "upstream_only_state": upstream_only,
            "upstream_only_state_inactive": upstream_only_inactive,
            "mapped_parameters": mapped_parameters,
            "mapped_buffers": sum(name in dict(local.named_buffers()) for name in mapping),
            "buffer_max_abs": buffer_max_abs,
            "active_parameter_gradients": seen, "expected_parameter_gradients": expected,
            "gradient_activity": activity, "serialization": serial,
            "preprocessing": {"max_abs": preprocessing_error}, "report": report.to_dict()}


def _case_gwnet(cls: type[torch.nn.Module], batch: int, adjacency: np.ndarray) -> dict[str, object]:
    seq, pred, nodes = 12, 3, adjacency.shape[0]
    supports = list(adj_to_supports(adjacency))
    kwargs = dict(num_nodes=nodes, dropout=0.2, supports=supports, gcn_bool=True,
                  addaptadj=True, aptinit=None, in_dim=3, out_dim=pred,
                  residual_channels=4, dilation_channels=4, skip_channels=8,
                  end_channels=12, kernel_size=2, blocks=1, layers=2)
    torch.manual_seed(SEED); upstream = cls(**kwargs)
    torch.manual_seed(SEED + 1); local = LocalGWNet(**kwargs)
    values, marks = torch.randn(batch, seq, nodes), _marks(batch, seq, nodes)
    history = to_spatiotemporal(values, marks)
    args = (history, None, 0, 0, False)
    wrapper = GWNetWrapper(seq, pred, nodes, adjacency, 3, 0.2, 4, 4, 8, 12, 2, 1, 2)
    wrapper.net.load_state_dict(upstream.state_dict(), strict=True)
    local.load_state_dict(upstream.state_dict(), strict=True); local.eval(); wrapper.eval()
    with torch.no_grad():
        error = float((wrapper(values, marks) - local(history, None, 0, 0, False)[..., 0]).abs().max())
    return _common_finish(local, upstream, wrapper, args, (values, marks),
                          {"gconv.0": "gconv.0", "end_conv_2": "end_conv_2"}, error)


def _case_stnorm(cls: type[torch.nn.Module], batch: int, adjacency: np.ndarray) -> dict[str, object]:
    seq, pred, nodes = 12, 3, adjacency.shape[0]
    kwargs = dict(num_nodes=nodes, tnorm_bool=True, snorm_bool=True, in_dim=3,
                  out_dim=pred, channels=4, kernel_size=2, blocks=1, layers=2)
    torch.manual_seed(SEED); upstream = cls(**kwargs)
    torch.manual_seed(SEED + 1); local = LocalSTNorm(**kwargs)
    values, marks = torch.randn(batch, seq, nodes), _marks(batch, seq, nodes)
    history = to_spatiotemporal(values, marks)
    args = (history, None, 0, 0, False)
    mapping, _ = _state_map(local, upstream); local.load_state_dict({k: upstream.state_dict()[v] for k, v in mapping.items()})
    wrapper = STNormWrapper(seq, pred, nodes, adjacency, 3, 4, 2, 1, 2, True, True)
    wrapper.net.load_state_dict(local.state_dict(), strict=True); local.eval(); wrapper.eval()
    with torch.no_grad():
        error = float((wrapper(values, marks) - local(history, None, 0, 0, False)[..., 0]).abs().max())
    return _common_finish(local, upstream, wrapper, args, (values, marks),
                          {"tn.0": "tn.0", "sn.0": "sn.0", "end_conv_2": "end_conv_2"}, error)


def _stdn_args(nodes: int) -> dict[str, object]:
    return {"Data": {"num_of_vertices": nodes, "time_slice_size": 60, "dataset_name": "modern_tsf"},
            "Training": {"L": 1, "K": 2, "d": 2, "node_miss_rate": 0.0,
                         "T_miss_len": 0, "order": 2, "reference": 2, "num_his": 12,
                         "num_pred": 12, "in_channels": 1, "out_channels": 1}}


def _case_stdn(cls: type[torch.nn.Module], batch: int, adjacency: np.ndarray) -> dict[str, object]:
    # The pinned STDN source adds its spatial embedding to the history block
    # while constructing that embedding with ``num_pred`` repeats. Therefore
    # the exact upstream contract requires num_his == num_pred.
    seq, pred, nodes = 12, 12, adjacency.shape[0]
    kwargs = _stdn_args(nodes)
    torch.manual_seed(SEED); upstream = cls(kwargs, bn_decay=0.1)
    torch.manual_seed(SEED + 1); local = LocalSTDN(kwargs, bn_decay=0.1)
    values, marks = torch.randn(batch, seq, nodes), _marks(batch, seq, nodes)
    future_marks = _marks(batch, pred, nodes)
    wrapper = STDNWrapper(seq, pred, nodes, adjacency, 60, 2, 2, 1, 2, 2, 1)
    st = to_spatiotemporal(values, marks)
    te = wrapper._build_te(marks, future_marks, st)
    args = (st[..., :1], te, wrapper.lpls, "test")
    wrapper.net.load_state_dict(upstream.state_dict(), strict=True)
    local.load_state_dict(upstream.state_dict(), strict=True); local.eval(); wrapper.eval()
    with torch.no_grad():
        error = float((wrapper(values, marks, x_mark_dec=future_marks) - local(*args)[..., 0]).abs().max())
    return _common_finish(local, upstream, wrapper, args, (values, marks, None, future_marks),
                          {"TEmbedding": "TEmbedding", "GCN": "GCN", "FC_2": "FC_2"}, error)


CASES = {"GWNet": _case_gwnet, "STNorm": _case_stnorm, "STDN": _case_stdn}


def verify_model(name: str, cls: type[torch.nn.Module]) -> dict[str, object]:
    nontrivial = np.array([[1, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 1], [0, 0, 1, 1]], dtype=np.float32)
    cases = {
        "batch_one_identity": CASES[name](cls, 1, np.eye(4, dtype=np.float32)),
        "batch_two_nontrivial_graph": CASES[name](cls, 2, nontrivial),
    }
    return {"schema_version": 1, "model": name, "passed": all(case["passed"] for case in cases.values()),
            "source": {"url": SOURCE_URL, "revision": SOURCE_REVISION, "license": SOURCE_LICENSE,
                       "license_sha256": LICENSE_SHA256, "files": SOURCE_FILES[name]},
            "upstream_execution": "exact-pinned-checkout", "mapping_version": "remaining-basicts-v1",
            "command": "uv run python scripts/verify_remaining_basicts_parity.py --upstream-checkout <BasicTS@c218c07b>",
            "tolerances": {"atol": 1e-6, "rtol": 1e-5}, "cases": cases}


def _errors(detail: dict[str, object], group: str) -> tuple[float, float]:
    items = [item for case in detail["cases"].values() for mode in case["report"]["modes"].values()
             for item in mode[group].values()]
    return max(float(x["max_abs"]) for x in items), max(float(x["max_rel"]) for x in items)


def _check(passed: bool, evidence: list[str], **metrics: float | int | str) -> dict[str, object]:
    return {"passed": passed, "evidence": evidence, "metrics": metrics}


def canonical_result(name: str, detail: dict[str, object], path: Path) -> dict[str, object]:
    fields = next(item for item in model_records(ROOT) if item["name"] == name)
    relative = path.relative_to(ROOT).as_posix()
    evidence = [relative, "tests/test_remaining_basicts_parity.py"]
    errors = {group: _errors(detail, group) for group in
              ("outputs", "intermediates", "input_gradients", "parameter_gradients")}
    first = next(iter(detail["cases"].values()))
    passed = bool(detail["passed"])
    serial = all(item[0] for case in detail["cases"].values() for item in case["serialization"].values())
    return {"schema_version": 1, "kind": "upstream-parity", "implementation": "upstream", "model": name,
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "subject_sha256": verification_subject_sha256(ROOT, fields),
            "artifacts": {relative: _sha(path)},
            "commands": [detail["command"], "uv run python -m unittest tests.test_remaining_basicts_parity -v",
                         f"uv run tsf repo doctor --strict --models {name}"],
            "environment": {"python": platform.python_version(), "framework": f"torch {torch.__version__}",
                "dependencies": {"numpy": np.__version__, "torch": torch.__version__}, "platform": platform.platform(),
                "device": "cpu", "dtype": "float32",
                "deterministic": {"seed": SEED, "algorithms": True, "num_threads": 1}},
            "passed": passed, "source": {"url": SOURCE_URL, "revision": SOURCE_REVISION, "license": SOURCE_LICENSE},
            "mapping": {"version": "remaining-basicts-v1", "parameters": first["mapped_parameters"],
                        "buffers": first["mapped_buffers"]},
            "fixture": {"identifier": "remaining-basicts-identity-nontrivial-v1",
                        "description": "CPU float32 batch=1/2 identity and nontrivial graph cases with calendar covariates."},
            "tolerances": detail["tolerances"], "modes": ["eval", "train"],
            "checks": {
                "outputs": _check(passed, evidence, max_abs=errors["outputs"][0], max_rel=errors["outputs"][1]),
                "intermediates": _check(passed, evidence, max_abs=errors["intermediates"][0], max_rel=errors["intermediates"][1]),
                "input_gradients": _check(passed, evidence, max_abs=errors["input_gradients"][0], max_rel=errors["input_gradients"][1]),
                "parameter_gradients": _check(passed, evidence, max_abs=errors["parameter_gradients"][0], max_rel=errors["parameter_gradients"][1]),
                "train_eval": _check(passed, evidence, modes="eval,train"),
                "buffers": _check(
                    all(case["buffer_max_abs"] == 0.0 for case in detail["cases"].values()),
                    evidence, mapped_buffers=first["mapped_buffers"],
                    max_abs=max(case["buffer_max_abs"] for case in detail["cases"].values()),
                ),
                "serialization": _check(serial, evidence, max_abs=0.0),
                "preprocessing": _check(passed, evidence, contract="calendar marks and adjacency/support or Laplacian inputs match exact backbone inputs"),
                "boundaries": _check(passed, evidence, cases="batch_one_identity,batch_two_nontrivial_graph"),
            }}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-checkout", required=True, type=Path)
    parser.add_argument("--models", nargs="*", choices=sorted(CASES), default=sorted(CASES))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "verification" / "parity")
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True); torch.set_num_threads(1)
    classes = _load_upstream(args.upstream_checkout.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    passed = True
    for name in args.models:
        detail = verify_model(name, classes[name])
        output = args.output_dir / f"{name}.json"
        output.write_text(json.dumps(detail, indent=2) + "\n", encoding="utf-8")
        if detail["passed"]:
            write_verification_result(ROOT / DEFAULT_INDEX, canonical_result(name, detail, output))
        print(f"{'PASS' if detail['passed'] else 'FAIL'} {name}: {output}")
        passed &= bool(detail["passed"])
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
