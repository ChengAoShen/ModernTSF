#!/usr/bin/env python3
"""Exact pinned-source parity for STID and MoFo.

xPatch and Pyraformer are deliberately not emitted by this harness: their
documented default-path blockers require either changing pinned source behavior
or changing the repository input contract, neither of which is parity.
"""

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
from types import ModuleType, SimpleNamespace

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
from components.marks import to_spatiotemporal  # noqa: E402
from models.mofo.model import Model as LocalMoFo  # noqa: E402
from models.stid.model import Model as LocalSTID  # noqa: E402


SOURCES = {
    "STID": {
        "url": "https://github.com/GestaltCogTeam/BasicTS",
        "revision": "c218c07b6ce5e4cf908b147fd180c486346fed9c",
        "license": "Apache-2.0",
        "license_sha256": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
        "file": "baselines/STID/arch/stid_arch.py",
        "file_sha256": "5a6dc9ade1a9fb57174aa35070d06d632d93033f7b7190d0df25ef036531d080",
    },
    "MoFo": {
        "url": "https://github.com/PoorOtterBob/MoFo",
        "revision": "2d14b47ea839c3809952b412340d72393f2521dc",
        "license": "MIT",
        "license_sha256": "212e9147f810f6ca6d14f3f6b0182f141e69f1da7d2c429aaebf27e6c0ed25b6",
        "file": "ts_benchmark/baselines/time_series_library/patchs/MoFo.py",
        "file_sha256": "3f87d18200975a694a533ac96114f4b185227cf5eadf3f941c1e84cf2d14b313",
    },
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_source(name: str, checkout: Path) -> Path:
    source = SOURCES[name]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=checkout, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if revision != source["revision"]:
        raise ValueError(f"{name} checkout is {revision}, expected {source['revision']}")
    if _sha(checkout / "LICENSE") != source["license_sha256"]:
        raise ValueError(f"{name} license digest mismatch")
    path = checkout / source["file"]
    if _sha(path) != source["file_sha256"]:
        raise ValueError(f"{name} source digest mismatch")
    return path


def _load_stid(checkout: Path):
    path = _check_source("STID", checkout)
    saved = {k: v for k, v in sys.modules.items() if k == "pinned_stid" or k.startswith("pinned_stid.")}
    for key in saved:
        del sys.modules[key]
    package = ModuleType("pinned_stid")
    package.__path__ = [str(path.parent)]  # type: ignore[attr-defined]
    sys.modules["pinned_stid"] = package
    try:
        spec = importlib.util.spec_from_file_location("pinned_stid.stid_arch", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.STID
    finally:
        for key in list(sys.modules):
            if key == "pinned_stid" or key.startswith("pinned_stid."):
                del sys.modules[key]
        sys.modules.update(saved)


def _load_mofo(checkout: Path):
    path = _check_source("MoFo", checkout)
    spec = importlib.util.spec_from_file_location("pinned_mofo", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.MoFo


def _round_trip(model, args):
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
    return {"passed": bool(torch.equal(expected, actual)), "max_abs": float((expected - actual).abs().max())}


def _raw_marks(batch: int, length: int) -> torch.Tensor:
    marks = torch.zeros(batch, length, 6)
    steps = torch.arange(length)
    marks[..., 3] = (steps // 24) % 7
    marks[..., 4] = steps % 24
    return marks


def _activity(model, args, seed):
    model.zero_grad(set_to_none=True)
    cloned = tuple(x.detach().clone().requires_grad_(x.is_floating_point()) if torch.is_tensor(x) else x for x in args)
    torch.manual_seed(seed)
    model(*cloned).float().sum().backward()
    return sorted(name for name, parameter in model.named_parameters() if parameter.grad is not None)


def _case_stid(upstream_cls, batch):
    seed, seq, pred, nodes, dim = 7331, 12, 6, 4, 8
    upstream_args = dict(num_nodes=nodes, node_dim=dim, input_len=seq, input_dim=3,
        embed_dim=dim, output_len=pred, num_layer=1, temp_dim_tid=dim,
        temp_dim_diw=dim, time_of_day_size=24, day_of_week_size=7,
        if_T_i_D=True, if_D_i_W=True, if_node=True)
    torch.manual_seed(seed); upstream = upstream_cls(**upstream_args)
    torch.manual_seed(seed + 1); wrapper = LocalSTID(seq, pred, nodes, input_dim=3,
        embed_dim=dim, num_layers=1, num_time_in_day=24, num_day_in_week=7)
    local = wrapper.net
    mapping = {name: name for name in local.state_dict()}
    values, marks = torch.randn(batch, seq, nodes), _raw_marks(batch, seq)
    history = to_spatiotemporal(values, marks)
    args = (history, None, 0, 0, False)
    modules = {"time_series_emb_layer": "time_series_emb_layer", "encoder.0": "encoder.0", "regression_layer": "regression_layer"}
    report = compare_model_parity(local, upstream, args, state_map=mapping,
        module_map=modules, seed=seed, atol=1e-6, rtol=1e-5)
    wrapper.net.load_state_dict(upstream.state_dict(), strict=True)
    wrapper.eval(); upstream.eval()
    with torch.no_grad():
        wrapped = wrapper(values, marks)
        direct = upstream(history, None, 0, 0, False)[..., 0]
    pre_error = float((wrapped - direct).abs().max())
    activity = {mode: (_activity(local.train(mode == "train"), args, seed), _activity(upstream.train(mode == "train"), args, seed)) for mode in ("eval", "train")}
    serial = {"local": _round_trip(local, args), "upstream": _round_trip(upstream, args), "wrapper": _round_trip(wrapper, (values, marks))}
    passed = report.passed and pre_error == 0.0 and all(a == b for a, b in activity.values()) and all(x["passed"] for x in serial.values())
    return {"passed": passed, "batch": batch, "state_map": mapping, "mapped_buffers": len(dict(local.named_buffers())), "activity": activity, "serialization": serial, "preprocessing": {"max_abs": pre_error, "contract": "raw six-column marks to BasicTS value/time-of-day/day-of-week history"}, "report": report.to_dict()}


def _case_mofo(upstream_cls, batch):
    seed, seq, pred, channels, dim = 7331, 48, 24, 3, 8
    config = SimpleNamespace(task_name="long_term_forecast", seq_len=seq,
        pred_len=pred, enc_in=channels, d_model=dim, periodic=24, head=2,
        d_layers=1, bias=1, cias=1)
    torch.manual_seed(seed); upstream = upstream_cls(config)
    torch.manual_seed(seed + 1); wrapper = LocalMoFo(seq, pred, channels, dim, 24, 2, 1, 1, 1)
    local = wrapper.net
    mapping = {name: name for name in local.state_dict() if name in upstream.state_dict()}
    if set(mapping) != set(local.state_dict()):
        raise ValueError("MoFo local state contains an unmapped entry")
    values, marks = torch.randn(batch, seq, channels), _raw_marks(batch, seq)
    synth = wrapper._build_marks(marks)
    args = (values, synth, None, None, None)
    modules = {"input": "input", "MoFo_Backbone.0.attn": "MoFo_Backbone.0.attn", "output": "output"}
    report = compare_model_parity(local, upstream, args, state_map=mapping,
        module_map=modules, seed=seed, atol=1e-6, rtol=1e-5)
    local.load_state_dict({name: upstream.state_dict()[name] for name in mapping}, strict=True)
    wrapper.eval(); upstream.eval()
    with torch.no_grad():
        wrapped = wrapper(values, marks)
        direct = upstream(values, synth, None, None, None)
    pre_error = float((wrapped - direct).abs().max())
    activity = {mode: (_activity(local.train(mode == "train"), args, seed), _activity(upstream.train(mode == "train"), args, seed)) for mode in ("eval", "train")}
    # Upstream-only task-branch parameters are intentionally inactive in forecast.
    mapped_upstream_activity = {mode: sorted(name for name in right if name in mapping.values()) for mode, (_, right) in activity.items()}
    activity_match = all(left == mapped_upstream_activity[mode] for mode, (left, _) in activity.items())
    serial = {"local": _round_trip(local, args), "upstream": _round_trip(upstream, args), "wrapper": _round_trip(wrapper, (values, marks))}
    passed = report.passed and pre_error == 0.0 and activity_match and all(x["passed"] for x in serial.values())
    return {"passed": passed, "batch": batch, "state_map": mapping, "mapped_buffers": len(dict(local.named_buffers())), "activity": activity, "upstream_only_inactive": sorted(set(upstream.state_dict()) - set(mapping.values())), "serialization": serial, "preprocessing": {"max_abs": pre_error, "contract": "raw marks converted to exact upstream TFB-normalized periodic-position columns"}, "report": report.to_dict()}


def _errors(detail, group):
    values = [item for case in detail["cases"].values() for mode in case["report"]["modes"].values() for item in mode[group].values()]
    return max(float(x["max_abs"]) for x in values), max(float(x["max_rel"]) for x in values)


def _check(passed, evidence, **metrics):
    return {"passed": passed, "evidence": evidence, "metrics": metrics}


def _canonical(name, detail, path):
    fields = next(item for item in model_records(ROOT) if item["name"] == name)
    relative = path.relative_to(ROOT).as_posix()
    evidence = [relative, "scripts/verify_special_upstream_parity.py", "tests/test_special_upstream_parity.py"]
    errors = {group: _errors(detail, group) for group in ("outputs", "intermediates", "input_gradients", "parameter_gradients")}
    first = next(iter(detail["cases"].values()))
    passed = bool(detail["passed"])
    serial = all(x["passed"] for case in detail["cases"].values() for x in case["serialization"].values())
    source = SOURCES[name]
    return {"schema_version": 1, "kind": "upstream-parity", "implementation": "upstream", "model": name,
        "verified_at": datetime.now(timezone.utc).isoformat(), "subject_sha256": verification_subject_sha256(ROOT, fields),
        "artifacts": {relative: _sha(path), "scripts/verify_special_upstream_parity.py": _sha(ROOT / "scripts/verify_special_upstream_parity.py"), "tests/test_special_upstream_parity.py": _sha(ROOT / "tests/test_special_upstream_parity.py")},
        "commands": [detail["command"], "uv run python -m unittest tests.test_special_upstream_parity -v", f"uv run tsf repo doctor --strict --models {name}"],
        "environment": {"python": platform.python_version(), "framework": f"torch {torch.__version__}", "dependencies": {"numpy": np.__version__, "torch": torch.__version__}, "platform": platform.platform(), "device": "cpu", "dtype": "float32", "deterministic": {"seed": 7331, "algorithms": True, "num_threads": 1}},
        "passed": passed, "source": {key: source[key] for key in ("url", "revision", "license")},
        "mapping": {"version": "special-upstream-v1", "parameters": len(first["state_map"]), "buffers": first["mapped_buffers"]},
        "fixture": {"identifier": "special-upstream-batch-boundaries-v1", "description": "Seeded CPU float32 batch=1/2 cases cover eval/train, defining intermediates, all active gradients, serialization, and preprocessing."},
        "tolerances": detail["tolerances"], "modes": ["eval", "train"],
        "checks": {**{group: _check(passed, evidence, max_abs=value[0], max_rel=value[1]) for group, value in errors.items()},
            "train_eval": _check(passed, evidence, modes="eval,train"), "buffers": _check(passed, evidence, mapped_buffers=first["mapped_buffers"]),
            "serialization": _check(serial, evidence, max_abs=0.0), "preprocessing": _check(passed, evidence, contract=first["preprocessing"]["contract"]),
            "boundaries": _check(passed, evidence, cases=",".join(detail["cases"]))}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basicts-checkout", required=True, type=Path)
    parser.add_argument("--mofo-checkout", required=True, type=Path)
    parser.add_argument("--models", nargs="*", choices=sorted(SOURCES), default=sorted(SOURCES))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "verification" / "parity")
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True); torch.set_num_threads(1)
    loaders = {"STID": lambda: _load_stid(args.basicts_checkout.resolve()), "MoFo": lambda: _load_mofo(args.mofo_checkout.resolve())}
    functions = {"STID": _case_stid, "MoFo": _case_mofo}
    command = "uv run python scripts/verify_special_upstream_parity.py --basicts-checkout <BasicTS@c218c07b> --mofo-checkout <MoFo@2d14b47e>"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    passed = True
    for name in args.models:
        cases = {f"batch_{batch}": functions[name](loaders[name](), batch) for batch in (1, 2)}
        detail = {"schema_version": 1, "model": name, "passed": all(case["passed"] for case in cases.values()), "source": SOURCES[name], "upstream_execution": "exact-pinned-checkout", "mapping_version": "special-upstream-v1", "command": command, "tolerances": {"atol": 1e-6, "rtol": 1e-5}, "cases": cases}
        output = args.output_dir / f"{name}.json"
        output.write_text(json.dumps(detail, indent=2) + "\n", encoding="utf-8")
        if detail["passed"]:
            write_verification_result(ROOT / DEFAULT_INDEX, _canonical(name, detail, output))
        print(f"{'PASS' if detail['passed'] else 'FAIL'} {name}: {output}")
        passed &= detail["passed"]
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
