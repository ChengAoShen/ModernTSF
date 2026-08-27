#!/usr/bin/env python3
"""Verify STAEformer, StemGNN, and TimeBridge against pinned sources."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
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
from models.staeformer.model import Model as LocalSTAEWrapper, STAEformer as LocalSTAE  # noqa: E402
from models.stemgnn.model import Model as LocalStemWrapper, StemGNN as LocalStem  # noqa: E402
from models.timebridge.model import Model as LocalTimeBridge  # noqa: E402


SOURCES = {
    "STAEformer": {
        "url": "https://github.com/GestaltCogTeam/BasicTS",
        "revision": "c218c07b6ce5e4cf908b147fd180c486346fed9c",
        "license": "Apache-2.0",
        "license_sha256": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
        "files": {"baselines/STAEformer/arch/staeformer_arch.py": "27cdb0d7333789075652a5b6cf30b57c4b691b4a1f1470863fbb98636006f0a4"},
    },
    "StemGNN": {
        "url": "https://github.com/GestaltCogTeam/BasicTS",
        "revision": "c218c07b6ce5e4cf908b147fd180c486346fed9c",
        "license": "Apache-2.0",
        "license_sha256": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
        "files": {"baselines/StemGNN/arch/stemgnn_arch.py": "e8d5a3833064c6133b8e6aeb52cc99547ef11f59d2af99c8183dfacc6f9b9c50"},
    },
    "TimeBridge": {
        "url": "https://github.com/Hank0626/TimeBridge",
        "revision": "0f9a83fbc3e1260c9ddd527c522dff0ce4b9554b",
        "license": "MIT",
        "license_sha256": "475078cc1f6de41e1b39c66b61aad6e870ddec14ce7132e18241ed74b9b3b6ff",
        "files": {
            "model/TimeBridge.py": "85b2ba4e7f3199c3dafc9128296c248b02ec53335ac1f5d155e8a445b910a4b5",
            "layers/Embed.py": "ca5b1221b4364577a75bb4794014dc6320849a70ad59fa8355c39fcc3eb6f7d2",
            "layers/SelfAttention_Family.py": "90b333e78699d9caeade7aae1f880f54607292a1e470ae9ad28adc4ed8776b26",
            "layers/Transformer_EncDec.py": "8c60fde6e9e5ff47f9524cb7d7ce2cbc80f6f4f706b9fc32bfd56d9120b49dc7",
            "layers/utils.py": "25f516891eb0537a828689b895205e303ed1ffe40428c682b44d3bf587bda866",
        },
    },
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_checkout(name: str, checkout: Path) -> None:
    source = SOURCES[name]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=checkout, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if revision != source["revision"]:
        raise ValueError(f"{name} checkout is {revision}, expected {source['revision']}")
    if _sha(checkout / "LICENSE") != source["license_sha256"]:
        raise ValueError(f"{name} LICENSE digest mismatch")
    for relative, expected in source["files"].items():
        if _sha(checkout / relative) != expected:
            raise ValueError(f"{name} source digest mismatch for {relative}")


def _load_basicts(checkout: Path) -> dict[str, type[torch.nn.Module]]:
    _verify_checkout("STAEformer", checkout)
    for key in tuple(sys.modules):
        if key == "baselines" or key.startswith("baselines.") or key == "basicts" or key.startswith("basicts."):
            del sys.modules[key]
    package = ModuleType("basicts")
    package.__path__ = [str(checkout / "basicts")]
    sys.modules["basicts"] = package
    sys.path.insert(0, str(checkout))
    try:
        return {
            "STAEformer": importlib.import_module("baselines.STAEformer.arch.staeformer_arch").STAEformer,
            "StemGNN": importlib.import_module("baselines.StemGNN.arch.stemgnn_arch").StemGNN,
        }
    finally:
        sys.path.remove(str(checkout))


def _load_timebridge(checkout: Path) -> type[torch.nn.Module]:
    _verify_checkout("TimeBridge", checkout)
    for key in tuple(sys.modules):
        if key == "model" or key.startswith("model.") or key == "layers" or key.startswith("layers."):
            del sys.modules[key]
    sys.path.insert(0, str(checkout))
    try:
        return importlib.import_module("model.TimeBridge").Model
    finally:
        sys.path.remove(str(checkout))


def _state_map(local: torch.nn.Module, upstream: torch.nn.Module) -> dict[str, str]:
    left, right = local.state_dict(), upstream.state_dict()
    missing = sorted(set(left) - set(right))
    if missing:
        raise ValueError(f"local state absent upstream: {missing}")
    for name, value in left.items():
        if value.shape != right[name].shape:
            raise ValueError(f"state shape differs for {name}")
    return {name: name for name in left}


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


def _active(model: torch.nn.Module, args: tuple[object, ...], seed: int) -> set[str]:
    cloned = tuple(
        value.detach().clone().requires_grad_(value.is_floating_point())
        if torch.is_tensor(value) else value for value in args
    )
    model.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    model(*cloned).float().sum().backward()
    return {name for name, parameter in model.named_parameters() if parameter.grad is not None}


def _activity(local, upstream, args, mapping, seed) -> dict[str, object]:
    modes = {}
    for mode in ("eval", "train"):
        local.train(mode == "train"); upstream.train(mode == "train")
        left = _active(local, args, seed)
        right = _active(upstream, args, seed)
        expected = {name for name, upstream_name in mapping.items() if upstream_name in right}
        modes[mode] = {"local": sorted(left), "upstream_mapped": sorted(expected), "matched": left == expected}
    return {"modes": modes, "active": sorted(set().union(*(set(item["local"]) for item in modes.values())))}


def _raw_marks(batch: int, length: int) -> torch.Tensor:
    marks = torch.zeros(batch, length, 6)
    marks[..., 0] = 2024
    marks[..., 1] = torch.where(torch.arange(length) < length // 2, 2, 3)
    marks[..., 2] = (torch.arange(length) % 27) + 1
    marks[..., 3] = torch.arange(length) % 7
    marks[..., 4] = torch.arange(length) % 24
    return marks


def _hourly_features(marks: torch.Tensor) -> torch.Tensor:
    year, month, day = marks[..., 0], marks[..., 1], marks[..., 2]
    weekday, hour = marks[..., 3], marks[..., 4]
    offsets = marks.new_tensor([0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    mi = month.long()
    leap = ((year.long() % 4 == 0) & ((year.long() % 100 != 0) | (year.long() % 400 == 0)))
    doy = offsets[mi] + day + (leap & (mi > 2)).to(day.dtype)
    return torch.stack((hour / 23 - .5, weekday / 6 - .5, (day - 1) / 30 - .5, (doy - 1) / 365 - .5), -1)


def _finish(report, mapping, local, upstream, args, preprocessing, batch, activity) -> dict[str, object]:
    serialization = {"local": _round_trip(local, args), "upstream": _round_trip(upstream, args)}
    gradients = min(len(mode.parameter_gradients) for mode in report.modes.values())
    expected = min(len(item["local"]) for item in activity["modes"].values())
    passed = report.passed and preprocessing == 0.0 and all(item[0] for item in serialization.values())
    passed = passed and gradients == expected and all(item["matched"] for item in activity["modes"].values())
    return {
        "passed": passed, "batch": batch, "state_map": mapping,
        "mapped_buffers": len(dict(local.named_buffers())),
        "active_parameter_gradients": gradients, "expected_parameter_gradients": expected,
        "gradient_activity": activity, "serialization": serialization,
        "preprocessing": {"max_abs": preprocessing}, "report": report.to_dict(),
    }


def _stae_case(upstream_cls, batch: int) -> dict[str, object]:
    seed, seq, pred, nodes = 5119, 12, 3, 4
    args = dict(num_nodes=nodes, in_steps=seq, out_steps=pred, steps_per_day=24,
        input_dim=3, output_dim=1, input_embedding_dim=8, tod_embedding_dim=4,
        dow_embedding_dim=4, spatial_embedding_dim=0, adaptive_embedding_dim=8,
        feed_forward_dim=16, num_heads=2, num_layers=1, dropout=.1, use_mixed_proj=True)
    torch.manual_seed(seed); upstream = upstream_cls(**args)
    torch.manual_seed(seed + 1); local = LocalSTAE(**args)
    mapping = _state_map(local, upstream)
    values, marks = torch.randn(batch, seq, nodes), _raw_marks(batch, seq)
    history = to_spatiotemporal(values, marks)
    call = (history, None, 0, 0, False)
    report = compare_model_parity(local, upstream, call, state_map=mapping,
        module_map={"input_proj": "input_proj", "attn_layers_t.0": "attn_layers_t.0", "output_proj": "output_proj"},
        seed=seed, atol=1e-6, rtol=1e-5)
    activity = _activity(local, upstream, call, mapping, seed)
    wrapper = LocalSTAEWrapper(seq, pred, nodes, None, 3, 24, 8, 4, 4, 0, 8, 16, 2, 1, .1, True)
    wrapper.net.load_state_dict(local.state_dict(), strict=True); wrapper.eval(); upstream.eval()
    with torch.no_grad():
        preprocessing = float((wrapper(values, marks) - upstream(history, None, 0, 0, False)[..., 0]).abs().max())
    result = _finish(report, mapping, local, upstream, call, preprocessing, batch, activity)
    result["serialization"]["wrapper"] = _round_trip(wrapper, (values, marks))
    result["passed"] = result["passed"] and result["serialization"]["wrapper"][0]
    return result


def _stem_case(upstream_cls, batch: int) -> dict[str, object]:
    seed, seq, pred, nodes = 5119, 12, 3, 4
    args = dict(units=nodes, stack_cnt=2, time_step=seq, multi_layer=2,
                horizon=pred, dropout_rate=.1, leaky_rate=.2)
    torch.manual_seed(seed); upstream = upstream_cls(**args)
    torch.manual_seed(seed + 1); local = LocalStem(**args)
    mapping = _state_map(local, upstream)
    values, marks = torch.randn(batch, seq, nodes), _raw_marks(batch, seq)
    history = to_spatiotemporal(values, marks)
    call = (history, None, 0, 0, False)
    report = compare_model_parity(local, upstream, call, state_map=mapping,
        module_map={"GRU": "GRU", "stock_block.0": "stock_block.0", "fc": "fc"},
        seed=seed, atol=1e-6, rtol=1e-5)
    activity = _activity(local, upstream, call, mapping, seed)
    wrapper = LocalStemWrapper(seq, pred, nodes, None, 3, 2, .1, .2)
    wrapper.net.load_state_dict(local.state_dict(), strict=True); wrapper.eval(); upstream.eval()
    with torch.no_grad():
        preprocessing = float((wrapper(values, marks) - upstream(history, None, 0, 0, False)[..., 0]).abs().max())
    result = _finish(report, mapping, local, upstream, call, preprocessing, batch, activity)
    result["serialization"]["wrapper"] = _round_trip(wrapper, (values, marks))
    result["passed"] = result["passed"] and result["serialization"]["wrapper"][0]
    return result


def _time_case(upstream_cls, batch: int) -> dict[str, object]:
    seed, seq, pred, channels = 5119, 12, 3, 7
    config = SimpleNamespace(revin=True, enc_in=channels, period=4, seq_len=seq,
        pred_len=pred, num_p=2, ia_layers=1, pd_layers=1, ca_layers=1,
        stable_len=3, d_model=8, n_heads=2, d_ff=16, attn_dropout=.1,
        dropout=0., activation="gelu")
    torch.manual_seed(seed); upstream = upstream_cls(config)
    torch.manual_seed(seed + 1); local = LocalTimeBridge(seq, pred, channels, 4, 2,
        1, 1, 1, 3, 8, 2, 16, .1, 0., "gelu", True)
    mapping = _state_map(local, upstream)
    values, raw_marks = torch.randn(batch, seq, channels), _raw_marks(batch, seq)
    official_marks = _hourly_features(raw_marks)
    call = (values, official_marks, None, None)
    report = compare_model_parity(local, upstream, call, state_map=mapping,
        module_map={"embedding.proj.0": "embedding.proj.0", "encoder.attn_layers.0": "encoder.attn_layers.0", "decoder.1": "decoder.1"},
        seed=seed, atol=1e-6, rtol=1e-5)
    activity = _activity(local, upstream, call, mapping, seed)
    local.eval(); upstream.eval()
    with torch.no_grad():
        preprocessing = float((local(values, raw_marks) - upstream(values, official_marks, None, None)).abs().max())
    result = _finish(report, mapping, local, upstream, call, preprocessing, batch, activity)
    result["serialization"]["raw_marks_wrapper"] = _round_trip(local, (values, raw_marks, None, None))
    result["passed"] = result["passed"] and result["serialization"]["raw_marks_wrapper"][0]
    return result


def verify_model(name: str, upstream_cls) -> dict[str, object]:
    fn = {"STAEformer": _stae_case, "StemGNN": _stem_case, "TimeBridge": _time_case}[name]
    cases = {"batch_one": fn(upstream_cls, 1), "batch_two": fn(upstream_cls, 2)}
    source = SOURCES[name]
    return {"schema_version": 1, "model": name, "passed": all(case["passed"] for case in cases.values()),
        "source": {key: source[key] for key in ("url", "revision", "license", "license_sha256", "files")},
        "upstream_execution": "exact-pinned-checkout", "mapping_version": "spatiotemporal-transformer-v1",
        "command": "uv run python scripts/verify_spatiotemporal_transformer_parity.py --basicts-checkout <BasicTS@c218c07b> --timebridge-checkout <TimeBridge@0f9a83fb>",
        "tolerances": {"atol": 1e-6, "rtol": 1e-5}, "cases": cases}


def _errors(detail, group):
    values = [item for case in detail["cases"].values() for mode in case["report"]["modes"].values() for item in mode[group].values()]
    return max(float(item["max_abs"]) for item in values), max(float(item["max_rel"]) for item in values)


def _check(passed, evidence, **metrics):
    return {"passed": passed, "evidence": evidence, "metrics": metrics}


def canonical_result(name: str, detail: dict[str, object], path: Path) -> dict[str, object]:
    fields = next(item for item in model_records(ROOT) if item["name"] == name)
    relative = path.relative_to(ROOT).as_posix()
    evidence = [relative, "tests/test_spatiotemporal_transformer_parity.py"]
    errors = {group: _errors(detail, group) for group in ("outputs", "intermediates", "input_gradients", "parameter_gradients")}
    first = next(iter(detail["cases"].values())); passed = bool(detail["passed"])
    serial = all(item[0] for case in detail["cases"].values() for item in case["serialization"].values())
    source = SOURCES[name]
    return {"schema_version": 1, "kind": "upstream-parity", "implementation": "upstream",
        "model": name, "verified_at": datetime.now(timezone.utc).isoformat(),
        "subject_sha256": verification_subject_sha256(ROOT, fields),
        "artifacts": {relative: _sha(path)}, "commands": [detail["command"],
            "uv run python -m unittest tests.test_spatiotemporal_transformer_parity -v",
            f"uv run tsf repo doctor --strict --models {name}"],
        "environment": {"python": platform.python_version(), "framework": f"torch {torch.__version__}",
            "dependencies": {"numpy": np.__version__, "torch": torch.__version__},
            "platform": platform.platform(), "device": "cpu", "dtype": "float32",
            "deterministic": {"seed": 5119, "algorithms": True, "num_threads": 1}},
        "passed": passed, "source": {key: source[key] for key in ("url", "revision", "license")},
        "mapping": {"version": "spatiotemporal-transformer-v1", "parameters": len(first["state_map"]), "buffers": first["mapped_buffers"]},
        "fixture": {"identifier": "spatiotemporal-transformer-batch-v1", "description": "CPU float32 batch=1/2 cases with calendar boundary values and exact upstream preprocessing."},
        "tolerances": detail["tolerances"], "modes": ["eval", "train"],
        "checks": {
            "outputs": _check(passed, evidence, max_abs=errors["outputs"][0], max_rel=errors["outputs"][1]),
            "intermediates": _check(passed, evidence, max_abs=errors["intermediates"][0], max_rel=errors["intermediates"][1]),
            "input_gradients": _check(passed, evidence, max_abs=errors["input_gradients"][0], max_rel=errors["input_gradients"][1]),
            "parameter_gradients": _check(passed, evidence, max_abs=errors["parameter_gradients"][0], max_rel=errors["parameter_gradients"][1]),
            "train_eval": _check(passed, evidence, modes="eval,train"),
            "buffers": _check(passed, evidence, mapped_buffers=first["mapped_buffers"]),
            "serialization": _check(serial, evidence, max_abs=0.0),
            "preprocessing": _check(passed, evidence, contract="raw calendar marks mapped to exact pinned-upstream backbone inputs"),
            "boundaries": _check(passed, evidence, cases="batch_one,batch_two"),
        }}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basicts-checkout", required=True, type=Path)
    parser.add_argument("--timebridge-checkout", required=True, type=Path)
    parser.add_argument("--models", nargs="*", choices=sorted(SOURCES), default=sorted(SOURCES))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "verification" / "parity")
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True); torch.set_num_threads(1)
    loaded = _load_basicts(args.basicts_checkout.resolve())
    if "TimeBridge" in args.models:
        loaded["TimeBridge"] = _load_timebridge(args.timebridge_checkout.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    passed = True
    for name in args.models:
        detail = verify_model(name, loaded[name])
        output = args.output_dir / f"{name}.json"
        output.write_text(json.dumps(detail, indent=2) + "\n", encoding="utf-8")
        if detail["passed"]:
            write_verification_result(ROOT / DEFAULT_INDEX, canonical_result(name, detail, output))
        print(f"{'PASS' if detail['passed'] else 'FAIL'} {name}: {output}")
        passed &= bool(detail["passed"])
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
