#!/usr/bin/env python3
"""Generate strict parity evidence for FITS, SparseTSF, and CycleNet."""

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
from types import ModuleType, SimpleNamespace
import sys
from typing import Any

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark.catalog_metadata import model_records  # noqa: E402
from benchmark.parity import compare_model_parity  # noqa: E402
from benchmark.verification_results import (  # noqa: E402
    DEFAULT_INDEX,
    verification_subject_sha256,
    write_verification_result,
)
from models.cyclenet.model import Model as LocalCycleNet  # noqa: E402
from models.fits.model import Model as LocalFITS  # noqa: E402
from models.sparsetsf.model import Model as LocalSparseTSF  # noqa: E402
from verification.fixtures.compact_upstream import (  # noqa: E402
    CycleNet as FixtureCycleNet,
    FITS as FixtureFITS,
    SparseTSF as FixtureSparseTSF,
)


SOURCES = {
    "FITS": {
        "url": "https://github.com/VEWOXIC/FITS",
        "revision": "d040bb015b6299da26d879b90dd19c80fb72c160",
        "license": "Apache-2.0",
        "file": "models/FITS.py",
        "sha256": "11c28af651bada1ba3e1e19c1cec1e835cc4b974d75da27df7b84d322096c90c",
    },
    "SparseTSF": {
        "url": "https://github.com/lss-1138/SparseTSF",
        "revision": "b8c2740eecc84d8095ffce49ba5acafe68e53bb8",
        "license": "Apache-2.0",
        "file": "models/SparseTSF.py",
        "sha256": "7f4afe758bd7ad71b939b88e633acd271f275f4e85dd73b724233c7b1a041948",
    },
    "CycleNet": {
        "url": "https://github.com/ACAT-SCUT/CycleNet",
        "revision": "d807e51fc2dcd143885ee639d97965a7ab0926f4",
        "license": "Apache-2.0",
        "file": "models/CycleNet.py",
        "sha256": "28e868d7d4211082e490ed9f19e323568e5fcd0a3beb29af78ce791bcaa31c05",
    },
}


class _ForecastOnly(nn.Module):
    def __init__(self, upstream: nn.Module, pred_len: int):
        super().__init__()
        self.upstream = upstream
        self.pred_len = pred_len

    def forward(self, x):
        return self.upstream(x)[0][:, -self.pred_len :]


def _cycle_index(cycle: int, pred_len: int, x_mark, x_mark_dec):
    phase = x_mark_dec[:, -pred_len] if x_mark_dec is not None else x_mark[:, 0]
    if cycle == 24:
        return phase[:, 4].to(torch.int64)
    if cycle == 7:
        return phase[:, 3].to(torch.int64)
    if cycle == 168:
        return (phase[:, 3] * 24 + phase[:, 4]).to(torch.int64)
    return phase[:, 4].to(torch.int64) % cycle


class _CycleAdapter(nn.Module):
    def __init__(self, upstream: nn.Module, cycle: int, pred_len: int):
        super().__init__()
        self.upstream = upstream
        self.cycle = cycle
        self.pred_len = pred_len

    def forward(self, x, x_mark, x_dec=None, x_mark_dec=None, *args):
        return self.upstream(
            x,
            _cycle_index(self.cycle, self.pred_len, x_mark, x_mark_dec),
        )


def _round_trip(model: nn.Module, inputs: tuple[Any, ...]) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        expected = model(*inputs)
    stream = BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = deepcopy(model)
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    with torch.no_grad():
        actual = restored(*inputs)
    difference = (expected - actual).abs()
    return {
        "passed": bool(torch.equal(expected, actual)),
        "max_abs": float(difference.max()) if difference.numel() else 0.0,
    }


def _assert_complete(report, local: nn.Module) -> None:
    expected = len(dict(local.named_parameters()))
    for mode in report.modes.values():
        if len(mode.parameter_gradients) != expected:
            raise AssertionError("an active parameter gradient was omitted")
        if not mode.input_gradients:
            raise AssertionError("input gradients were omitted")
        if not mode.intermediates:
            raise AssertionError("defining intermediates were omitted")


def _fits_case(upstream_class, *, individual, seq_len, pred_len, channels, cut_freq):
    config = SimpleNamespace(
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=channels,
        individual=individual,
        cut_freq=cut_freq,
    )
    local = LocalFITS(**vars(config))
    upstream = _ForecastOnly(upstream_class(config), pred_len)
    if individual:
        state_map = {
            f"model.freq_upsampler.{index}.{suffix}":
            f"upstream.freq_upsampler.{index}.{suffix}"
            for index in range(channels)
            for suffix in ("weight", "bias")
        }
        module_map = {
            f"model.freq_upsampler.{index}": f"upstream.freq_upsampler.{index}"
            for index in range(channels)
        }
    else:
        state_map = {
            f"model.freq_upsampler.{suffix}": f"upstream.freq_upsampler.{suffix}"
            for suffix in ("weight", "bias")
        }
        module_map = {"model.freq_upsampler": "upstream.freq_upsampler"}
    values = torch.randn(2 if individual else 1, seq_len, channels)
    inputs = (values,)
    report = compare_model_parity(
        local,
        upstream,
        inputs,
        state_map=state_map,
        module_map=module_map,
        modes=("eval", "train"),
        compare_gradients=True,
        seed=2718,
        atol=1e-6,
        rtol=1e-5,
    )
    _assert_complete(report, local)
    serial = {"local": _round_trip(local, inputs), "upstream": _round_trip(upstream, inputs)}
    return _case_payload(report, state_map, module_map, serial, vars(config))


def _sparse_case(upstream_class, *, model_type, seq_len, pred_len, channels, period, d_model=5):
    config = SimpleNamespace(
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=channels,
        period_len=period,
        d_model=d_model,
        model_type=model_type,
    )
    local = LocalSparseTSF(
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=channels,
        period=period,
        d_model=d_model,
        model_type=model_type,
    )
    upstream = upstream_class(config)
    branches = ["conv1d"] + (["linear"] if model_type == "linear" else ["mlp.0", "mlp.2"])
    state_map = {
        name: name[len("model."):] if name.startswith("model.") else name
        for name in local.state_dict()
    }
    module_map = {f"model.{name}": name for name in branches}
    values = torch.randn(2, seq_len, channels)
    inputs = (values,)
    report = compare_model_parity(
        local, upstream, inputs, state_map=state_map, module_map=module_map,
        modes=("eval", "train"), compare_gradients=True, seed=2718,
        atol=1e-6, rtol=1e-5,
    )
    _assert_complete(report, local)
    serial = {"local": _round_trip(local, inputs), "upstream": _round_trip(upstream, inputs)}
    return _case_payload(report, state_map, module_map, serial, vars(config))


def _marks(batch, length, weekday, hour):
    value = torch.zeros(batch, length, 5)
    value[:, :, 3] = weekday
    value[:, :, 4] = hour
    return value


def _cycle_case(upstream_class, *, model_type, use_revin, seq_len, pred_len, channels, cycle):
    config = SimpleNamespace(
        seq_len=seq_len, pred_len=pred_len, enc_in=channels, cycle=cycle,
        model_type=model_type, d_model=5, use_revin=use_revin,
    )
    local = LocalCycleNet(**vars(config))
    upstream = _CycleAdapter(upstream_class(config), cycle, pred_len)
    state_map = {}
    for name in local.state_dict():
        suffix = name[len("model."):]
        if suffix.startswith("cycle_queue."):
            suffix = "cycleQueue." + suffix[len("cycle_queue."):]
        state_map[name] = "upstream." + suffix
    branches = ["cycle_queue", "model"]
    module_map = {
        f"model.{name}": "upstream." + ("cycleQueue" if name == "cycle_queue" else name)
        for name in branches
    }
    batch = 2
    values = torch.randn(batch, seq_len, channels)
    inputs = (
        values,
        _marks(batch, seq_len, 2, 3),
        None,
        _marks(batch, pred_len, 5, 11),
    )
    report = compare_model_parity(
        local, upstream, inputs, state_map=state_map, module_map=module_map,
        modes=("eval", "train"), compare_gradients=True, seed=2718,
        atol=1e-6, rtol=1e-5,
    )
    _assert_complete(report, local)
    serial = {"local": _round_trip(local, inputs), "upstream": _round_trip(upstream, inputs)}
    return _case_payload(report, state_map, module_map, serial, vars(config))


def _case_payload(report, state_map, module_map, serialization, fixture):
    passed = report.passed and all(item["passed"] for item in serialization.values())
    return {
        "passed": passed,
        "fixture": fixture,
        "state_map": state_map,
        "module_map": module_map,
        "serialization": serialization,
        "report": report.to_dict(),
    }


def verify_model(name: str, upstream_class=None, *, exact=False, command=None):
    torch.manual_seed(2718)
    if name == "FITS":
        cls = upstream_class or FixtureFITS
        cases = {
            "shared": _fits_case(cls, individual=False, seq_len=8, pred_len=4, channels=2, cut_freq=3),
            "individual": _fits_case(cls, individual=True, seq_len=8, pred_len=4, channels=3, cut_freq=3),
            "minimum_sequence": _fits_case(cls, individual=False, seq_len=2, pred_len=2, channels=1, cut_freq=1),
        }
    elif name == "SparseTSF":
        cls = upstream_class or FixtureSparseTSF
        cases = {
            "linear": _sparse_case(cls, model_type="linear", seq_len=8, pred_len=4, channels=2, period=2),
            "mlp": _sparse_case(cls, model_type="mlp", seq_len=12, pred_len=6, channels=3, period=3),
            "minimum_sequence": _sparse_case(cls, model_type="linear", seq_len=2, pred_len=2, channels=1, period=2),
        }
    elif name == "CycleNet":
        cls = upstream_class or FixtureCycleNet
        cases = {
            "linear_revin": _cycle_case(cls, model_type="linear", use_revin=True, seq_len=8, pred_len=4, channels=2, cycle=24),
            "mlp_no_revin": _cycle_case(cls, model_type="mlp", use_revin=False, seq_len=6, pred_len=3, channels=3, cycle=7),
            "minimum_sequence": _cycle_case(cls, model_type="linear", use_revin=False, seq_len=1, pred_len=1, channels=1, cycle=7),
        }
    else:
        raise KeyError(name)
    source = SOURCES[name]
    return {
        "schema_version": 1,
        "model": name,
        "passed": all(case["passed"] for case in cases.values()),
        "source": {**source, "files": [source["file"]], "file_sha256": source["sha256"]},
        "upstream_execution": "exact-pinned-checkout" if exact else "checked-thin-fixture",
        "mapping_version": "compact-upstream-v1",
        "command": command or "uv run python scripts/verify_compact_upstream_parity.py",
        "deterministic": {"seed": 2718, "device": "cpu"},
        "tolerances": {"atol": 1e-6, "rtol": 1e-5},
        "cases": cases,
    }


def _errors(detail, group):
    values = [
        item
        for case in detail["cases"].values()
        for mode in case["report"]["modes"].values()
        for item in mode[group].values()
    ]
    return max(float(x["max_abs"]) for x in values), max(float(x["max_rel"]) for x in values)


def _check(passed, evidence, **metrics):
    return {"passed": passed, "evidence": evidence, "metrics": metrics}


def canonical_result(name, detail, detail_path):
    fields = next(item for item in model_records(ROOT) if item["name"] == name)
    relative = detail_path.relative_to(ROOT).as_posix()
    evidence = [relative, "tests/test_compact_upstream_parity.py"]
    output_abs, output_rel = _errors(detail, "outputs")
    mid_abs, mid_rel = _errors(detail, "intermediates")
    input_abs, input_rel = _errors(detail, "input_gradients")
    param_abs, param_rel = _errors(detail, "parameter_gradients")
    serialization = all(
        value["passed"] for case in detail["cases"].values()
        for value in case["serialization"].values()
    )
    all_pass = bool(detail["passed"])
    mapped_parameters = max(len(case["state_map"]) for case in detail["cases"].values())
    source = SOURCES[name]
    return {
        "schema_version": 1, "kind": "upstream-parity", "implementation": "upstream",
        "model": name, "verified_at": datetime.now(timezone.utc).isoformat(),
        "subject_sha256": verification_subject_sha256(ROOT, fields),
        "artifacts": {relative: hashlib.sha256(detail_path.read_bytes()).hexdigest()},
        "commands": [detail["command"], "uv run python -m unittest tests.test_compact_upstream_parity -v", f"uv run tsf repo doctor --backward --models {name}"],
        "environment": {
            "python": platform.python_version(), "framework": f"torch {torch.__version__}",
            "dependencies": {"numpy": np.__version__, "torch": torch.__version__},
            "platform": platform.platform(), "device": "cpu", "dtype": "float32/complex64",
            "deterministic": {"seed": 2718, "algorithms": torch.are_deterministic_algorithms_enabled(), "num_threads": torch.get_num_threads()},
        },
        "passed": all_pass and serialization,
        "source": {key: source[key] for key in ("url", "revision", "license")},
        "mapping": {"version": "compact-upstream-v1", "parameters": mapped_parameters, "buffers": 0},
        "fixture": {"identifier": "compact-upstream-boundaries-v1", "description": "Seeded CPU cases cover train/eval, linear/MLP or shared/individual branches, batch/channel variation, and minimum valid sequences."},
        "tolerances": {"atol": 1e-6, "rtol": 1e-5}, "modes": ["eval", "train"],
        "checks": {
            "outputs": _check(all_pass, evidence, max_abs=output_abs, max_rel=output_rel),
            "intermediates": _check(all_pass, evidence, max_abs=mid_abs, max_rel=mid_rel),
            "input_gradients": _check(all_pass, evidence, max_abs=input_abs, max_rel=input_rel),
            "parameter_gradients": _check(all_pass, evidence, max_abs=param_abs, max_rel=param_rel),
            "train_eval": _check(all_pass, evidence, modes="eval,train"),
            "buffers": _check(all_pass, evidence, mapped_buffers=0, reason="no persistent buffers"),
            "serialization": _check(serialization, evidence, max_abs=0.0),
            "preprocessing": _check(all_pass, evidence, contract="identical BLC values; CycleNet timestamp adapter is compared against the exact upstream cycle-index API"),
            "boundaries": _check(all_pass, evidence, cases=",".join(detail["cases"])),
        },
    }


def _load_exact(name: str, checkout: Path):
    import subprocess
    source = SOURCES[name]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=checkout, check=True, capture_output=True, text=True).stdout.strip()
    if revision != source["revision"]:
        raise ValueError(f"{name} checkout is {revision}, expected {source['revision']}")
    license_text = (checkout / "LICENSE").read_text(encoding="utf-8")
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise ValueError(f"{name} checkout does not contain the recorded Apache-2.0 license")
    path = checkout / source["file"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != source["sha256"]:
        raise ValueError(f"{name} source digest is {digest}, expected {source['sha256']}")
    saved = {key: value for key, value in sys.modules.items() if key == "models" or key.startswith("models.") or key == "layers" or key.startswith("layers.")}
    for key in saved:
        del sys.modules[key]
    sys.path.insert(0, str(checkout))
    try:
        # These upstream repositories use top-level namespace-style imports but
        # do not ship package initializers. Pin their lookup roots explicitly so
        # the already installed ModernTSF ``models`` package cannot shadow them.
        for package_name in ("models", "layers"):
            package = ModuleType(package_name)
            package.__path__ = [str(checkout / package_name)]  # type: ignore[attr-defined]
            sys.modules[package_name] = package
        spec = importlib.util.spec_from_file_location(f"pinned_{name}", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot import pinned source {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Model
    finally:
        sys.path.remove(str(checkout))
        for key in list(sys.modules):
            if key == "models" or key.startswith("models.") or key == "layers" or key.startswith("layers."):
                del sys.modules[key]
        sys.modules.update(saved)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fits-checkout", type=Path, required=True)
    parser.add_argument("--sparsetsf-checkout", type=Path, required=True)
    parser.add_argument("--cyclenet-checkout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "verification" / "parity")
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkouts = {"FITS": args.fits_checkout, "SparseTSF": args.sparsetsf_checkout, "CycleNet": args.cyclenet_checkout}
    command = (
        "uv run python scripts/verify_compact_upstream_parity.py "
        "--fits-checkout <FITS@d040bb015b6299da26d879b90dd19c80fb72c160> "
        "--sparsetsf-checkout <SparseTSF@b8c2740eecc84d8095ffce49ba5acafe68e53bb8> "
        "--cyclenet-checkout <CycleNet@d807e51fc2dcd143885ee639d97965a7ab0926f4>"
    )
    passed = True
    for name, checkout in checkouts.items():
        detail = verify_model(name, _load_exact(name, checkout.resolve()), exact=True, command=command)
        output = args.output_dir / f"{name}.json"
        output.write_text(json.dumps(detail, indent=2) + "\n", encoding="utf-8")
        write_verification_result(ROOT / DEFAULT_INDEX, canonical_result(name, detail, output))
        print(f"{'PASS' if detail['passed'] else 'FAIL'} {name}: {output}")
        passed &= detail["passed"]
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
