#!/usr/bin/env python3
"""Verify Transformer, Informer, and ETSformer against pinned TSLib sources."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import platform
import subprocess
import sys
from types import SimpleNamespace
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
from components.marks import adapt_tslib_marks  # noqa: E402
from models.etsformer.model import Model as LocalETSformer  # noqa: E402
from models.informer.model import Model as LocalInformer  # noqa: E402
from models.transformer.model import Model as LocalTransformer  # noqa: E402


SEED = 5441
SOURCES = {
    "Transformer": {
        "url": "https://github.com/thuml/Time-Series-Library",
        "revision": "2fb5b84ecef67c45a759f7cf82023d27afe27882",
        "license": "MIT",
        "license_sha256": "8a6caa178ea3f33ebff5d7bb5558628cf5b423305dc30d3630f86564c7db94a2",
        "files": {
            "models/Transformer.py": "46ded6c516fd03951bce0a82c3f4245f0247efd01fb5bde775145cfbb6d8435d",
            "layers/Embed.py": "17e7c3577324c41a0da427a199c955b782fde905aabb1f7cbc3c4e15ebd4ae35",
            "utils/timefeatures.py": "319b3da2ef15ccae95162b1b4f4a9a0b0da63fa0062a7f6006a06dd567258d85",
        },
    },
    "Informer": {
        "url": "https://github.com/thuml/Time-Series-Library",
        "revision": "2fb5b84ecef67c45a759f7cf82023d27afe27882",
        "license": "MIT",
        "license_sha256": "8a6caa178ea3f33ebff5d7bb5558628cf5b423305dc30d3630f86564c7db94a2",
        "files": {
            "models/Informer.py": "95dee85f6c0642dbfeee7736d526f9e20a54b5f5871a7085343b6a8be3cf56d2",
            "layers/Embed.py": "17e7c3577324c41a0da427a199c955b782fde905aabb1f7cbc3c4e15ebd4ae35",
            "utils/timefeatures.py": "319b3da2ef15ccae95162b1b4f4a9a0b0da63fa0062a7f6006a06dd567258d85",
        },
    },
    "ETSformer": {
        "url": "https://github.com/thuml/Time-Series-Library",
        "revision": "230805fe9f451b61e34b96116d995b417e343ac0",
        "license": "MIT",
        "license_sha256": "8a6caa178ea3f33ebff5d7bb5558628cf5b423305dc30d3630f86564c7db94a2",
        "files": {
            "models/ETSformer.py": "3cd25f1ea4dc18e926720b2976d86820363dcc5db0436e490f2b794630086035",
            "layers/ETSformer_EncDec.py": "072fd811d5dc3cf1698cf829c767c331d3f0dc341ef619e1424824bb7981c338",
            "layers/Embed.py": "596bf68383cc951c93e8e49df0994a0ce57ed41b2a90acb3c47e6c85dbf16a78",
            "utils/timefeatures.py": "2ad031a425f6f02b88db1918333a30937e41b18f435df90e6bcbb6470427221b",
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
        actual = _sha(checkout / relative)
        if actual != expected:
            raise ValueError(f"source digest mismatch for {relative}: {actual}")


def _load_upstream(name: str, checkout: Path) -> type[torch.nn.Module]:
    _verify_checkout(name, checkout)
    for module_name in list(sys.modules):
        if module_name == "layers" or module_name.startswith("layers."):
            del sys.modules[module_name]
    file_name = f"models/{name}.py"
    spec = importlib.util.spec_from_file_location(
        f"moderntsf_exact_{name.lower()}", checkout / file_name
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {checkout / file_name}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(checkout))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(checkout))
    return module.Model


class _RawMarksAdapter(torch.nn.Module):
    """Expose an exact pinned TSLib model through ModernTSF's raw-mark API."""

    def __init__(self, net: torch.nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.net(
            x_enc,
            adapt_tslib_marks(x_mark_enc, embed_type="timeF", freq="h"),
            x_dec,
            adapt_tslib_marks(x_mark_dec, embed_type="timeF", freq="h"),
            mask,
        )


def _raw_marks(batch: int, length: int, start: datetime) -> torch.Tensor:
    rows = []
    for offset in range(length):
        stamp = start + timedelta(hours=offset)
        rows.append(
            [stamp.year, stamp.month, stamp.day, stamp.weekday(), stamp.hour, stamp.minute]
        )
    return torch.tensor(rows, dtype=torch.float32).unsqueeze(0).expand(batch, -1, -1).clone()


def _official_hourly(marks: torch.Tensor) -> torch.Tensor:
    """Independent tensor expression for pinned ``utils.timefeatures``."""
    year, month, day = marks[..., 0], marks[..., 1], marks[..., 2]
    weekday, hour = marks[..., 3], marks[..., 4]
    offsets = marks.new_tensor([0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    month_index = month.long()
    leap = (year.long() % 4 == 0) & ((year.long() % 100 != 0) | (year.long() % 400 == 0))
    day_of_year = offsets[month_index] + day + (leap & (month_index > 2)).to(day.dtype)
    return torch.stack((
        hour / 23.0 - 0.5,
        weekday / 6.0 - 0.5,
        (day - 1.0) / 30.0 - 0.5,
        (day_of_year - 1.0) / 365.0 - 0.5,
    ), dim=-1)


def _round_trip(model: torch.nn.Module, args: tuple[object, ...]) -> tuple[bool, float]:
    model.eval()
    torch.manual_seed(SEED)
    with torch.no_grad():
        expected = model(*args)
    stream = BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = deepcopy(model)
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    restored.eval()
    torch.manual_seed(SEED)
    with torch.no_grad():
        actual = restored(*args)
    error = (expected - actual).abs()
    return torch.equal(expected, actual), float(error.max()) if error.numel() else 0.0


def _state_map(local: torch.nn.Module, upstream: _RawMarksAdapter, name: str) -> tuple[dict[str, str], list[str]]:
    left, right = local.state_dict(), upstream.state_dict()
    mapping = {}
    for local_name, tensor in left.items():
        candidate = f"net.{local_name}"
        if name == "Informer":
            candidate = candidate.replace(".down_conv.", ".downConv.")
        if candidate not in right or tensor.shape != right[candidate].shape:
            raise ValueError(f"unmapped or mismatched local state {local_name!r}")
        mapping[local_name] = candidate
    return mapping, sorted(set(right) - set(mapping.values()))


def _active(model: torch.nn.Module, args: tuple[object, ...], seed: int) -> set[str]:
    cloned = tuple(
        value.detach().clone().requires_grad_(value.is_floating_point())
        if torch.is_tensor(value) else value for value in args
    )
    model.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    model(*cloned).float().sum().backward()
    return {name for name, parameter in model.named_parameters() if parameter.grad is not None}


def _activity(local, upstream, args, mapping) -> dict[str, object]:
    modes = {}
    for mode in ("eval", "train"):
        local.train(mode == "train")
        upstream.train(mode == "train")
        left = _active(local, args, SEED)
        right = _active(upstream, args, SEED)
        expected = {local_name for local_name, upstream_name in mapping.items() if upstream_name in right}
        upstream_only_active = sorted(right - set(mapping.values()))
        modes[mode] = {
            "local": sorted(left),
            "upstream_mapped": sorted(expected),
            "upstream_only_active": upstream_only_active,
            "matched": left == expected and not upstream_only_active,
        }
    return {
        "modes": modes,
        "active": sorted(set().union(*(set(item["local"]) for item in modes.values()))),
    }


def _construct(name: str, upstream_cls: type[torch.nn.Module]):
    seq, pred, label, channels = 12, 3, 0, 3
    common = dict(
        task_name="long_term_forecast", seq_len=seq, pred_len=pred,
        label_len=label, enc_in=channels, c_out=channels, d_model=8,
        n_heads=2, e_layers=2, d_layers=2, d_ff=16, dropout=0.1,
        activation="gelu", embed="timeF", freq="h",
    )
    if name == "Transformer":
        upstream_cfg = {**common, "dec_in": channels, "factor": 5, "d_layers": 1}
        local = LocalTransformer(
            seq, pred, label, channels, d_model=8, n_heads=2, e_layers=2,
            d_layers=1, d_ff=16, dropout=0.1, activation="gelu",
            embed="timeF", freq="h",
        )
        modules = {
            "enc_embedding": "net.enc_embedding",
            "encoder.attn_layers.0": "net.encoder.attn_layers.0",
            "decoder.projection": "net.decoder.projection",
        }
    elif name == "Informer":
        upstream_cfg = {**common, "dec_in": channels, "factor": 3, "distil": True, "d_layers": 1}
        local = LocalInformer(
            seq, pred, label, channels, d_model=8, n_heads=2, e_layers=2,
            d_layers=1, d_ff=16, dropout=0.1, factor=3,
            activation="gelu", distil=True, embed="timeF", freq="h",
        )
        modules = {
            "enc_embedding": "net.enc_embedding",
            "encoder.conv_layers.0": "net.encoder.conv_layers.0",
            "decoder.projection": "net.decoder.projection",
        }
    else:
        upstream_cfg = {**common, "top_k": 2, "activation": "sigmoid"}
        local = LocalETSformer(
            seq, pred, channels, d_model=8, n_heads=2, e_layers=2,
            d_layers=2, d_ff=16, top_k=2, dropout=0.1,
            activation="sigmoid", embed="timeF", freq="h",
        )
        modules = {
            "enc_embedding": "net.enc_embedding",
            "encoder.layers.0.growth_layer": "net.encoder.layers.0.growth_layer",
            "decoder.pred": "net.decoder.pred",
        }
    torch.manual_seed(SEED)
    upstream = _RawMarksAdapter(upstream_cls(SimpleNamespace(**upstream_cfg)))
    return local, upstream, modules


def _case(name: str, upstream_cls: type[torch.nn.Module], batch: int) -> dict[str, object]:
    torch.manual_seed(SEED + 1)
    local, upstream, module_map = _construct(name, upstream_cls)
    mapping, upstream_only = _state_map(local, upstream, name)
    seq, pred, channels = 12, 3, 3
    values = torch.randn(batch, seq, channels)
    decoder = torch.randn(batch, pred, channels)
    marks = _raw_marks(batch, seq, datetime(2024, 2, 28, 18))
    decoder_marks = _raw_marks(batch, pred, datetime(2024, 3, 1, 0))
    args = (values, marks, decoder, decoder_marks, None)
    report = compare_model_parity(
        local, upstream, args, state_map=mapping, module_map=module_map,
        seed=SEED, atol=1e-6, rtol=1e-5,
    )
    activity = _activity(local, upstream, args, mapping)
    official_enc = _official_hourly(marks)
    official_dec = _official_hourly(decoder_marks)
    feature_error = max(
        float((adapt_tslib_marks(marks, embed_type="timeF", freq="h") - official_enc).abs().max()),
        float((adapt_tslib_marks(decoder_marks, embed_type="timeF", freq="h") - official_dec).abs().max()),
    )
    upstream_direct_args = (values, official_enc, decoder, official_dec, None)
    upstream.net.eval()
    local.eval()
    torch.manual_seed(SEED)
    with torch.no_grad():
        wrapped = upstream(*args)
    torch.manual_seed(SEED)
    with torch.no_grad():
        direct = upstream.net(*upstream_direct_args)
    wrapper_error = float((wrapped - direct).abs().max())
    serial = {
        "local_raw_marks": _round_trip(local, args),
        "upstream_raw_adapter": _round_trip(upstream, args),
        "upstream_preprocessed": _round_trip(upstream.net, upstream_direct_args),
    }
    gradients = min(len(mode.parameter_gradients) for mode in report.modes.values())
    expected = min(len(item["local"]) for item in activity["modes"].values())
    upstream_only_inactive = not any(item["upstream_only_active"] for item in activity["modes"].values())
    mapped_buffers = sum(name in dict(local.named_buffers()) for name in mapping)
    passed = (
        report.passed
        and feature_error == 0.0
        and wrapper_error == 0.0
        and all(item[0] for item in serial.values())
        and gradients == expected
        and all(item["matched"] for item in activity["modes"].values())
        and upstream_only_inactive
    )
    return {
        "passed": passed,
        "batch": batch,
        "state_map": mapping,
        "upstream_only_state": upstream_only,
        "upstream_only_state_inactive": upstream_only_inactive,
        "mapped_parameters": sum(name in dict(local.named_parameters()) for name in mapping),
        "mapped_buffers": mapped_buffers,
        "active_parameter_gradients": gradients,
        "expected_parameter_gradients": expected,
        "gradient_activity": activity,
        "serialization": serial,
        "preprocessing": {
            "feature_max_abs": feature_error,
            "wrapper_max_abs": wrapper_error,
        },
        "report": report.to_dict(),
    }


def verify_model(name: str, upstream_cls: type[torch.nn.Module]) -> dict[str, object]:
    cases = {
        "batch_one": _case(name, upstream_cls, 1),
        "batch_two": _case(name, upstream_cls, 2),
    }
    source = SOURCES[name]
    return {
        "schema_version": 1,
        "model": name,
        "passed": all(case["passed"] for case in cases.values()),
        "source": {key: source[key] for key in ("url", "revision", "license", "license_sha256", "files")},
        "upstream_execution": "exact-pinned-checkout",
        "mapping_version": "tslib-raw-hourly-marks-v1",
        "command": "uv run python scripts/verify_temporal_marks_parity.py --transformer-checkout <TSLib@2fb5b84e> --etsformer-checkout <TSLib@230805fe>",
        "tolerances": {"atol": 1e-6, "rtol": 1e-5},
        "cases": cases,
    }


def _errors(detail: dict[str, object], group: str) -> tuple[float, float]:
    values = [
        item
        for case in detail["cases"].values()
        for mode in case["report"]["modes"].values()
        for item in mode[group].values()
    ]
    return max(float(item["max_abs"]) for item in values), max(float(item["max_rel"]) for item in values)


def _check(passed: bool, evidence: list[str], **metrics) -> dict[str, object]:
    return {"passed": passed, "evidence": evidence, "metrics": metrics}


def canonical_result(name: str, detail: dict[str, object], path: Path) -> dict[str, object]:
    fields = next(item for item in model_records(ROOT) if item["name"] == name)
    relative = path.relative_to(ROOT).as_posix()
    evidence = [relative, "tests/test_temporal_marks_parity.py"]
    errors = {group: _errors(detail, group) for group in (
        "outputs", "intermediates", "input_gradients", "parameter_gradients"
    )}
    first = next(iter(detail["cases"].values()))
    passed = bool(detail["passed"])
    serial_error = max(
        float(result[1])
        for case in detail["cases"].values()
        for result in case["serialization"].values()
    )
    source = SOURCES[name]
    return {
        "schema_version": 1,
        "kind": "upstream-parity",
        "implementation": "upstream",
        "model": name,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "subject_sha256": verification_subject_sha256(ROOT, fields),
        "artifacts": {relative: _sha(path)},
        "commands": [
            detail["command"],
            "uv run python -m unittest tests.test_temporal_marks_parity -v",
            f"uv run tsf repo doctor --strict --models {name}",
        ],
        "environment": {
            "python": platform.python_version(),
            "framework": f"torch {torch.__version__}",
            "dependencies": {"numpy": np.__version__, "torch": torch.__version__},
            "platform": platform.platform(),
            "device": "cpu",
            "dtype": "float32",
            "deterministic": {"seed": SEED, "algorithms": True, "num_threads": 1},
        },
        "passed": passed,
        "source": {key: source[key] for key in ("url", "revision", "license")},
        "mapping": {
            "version": "tslib-raw-hourly-marks-v1",
            "parameters": first["mapped_parameters"],
            "buffers": first["mapped_buffers"],
        },
        "fixture": {
            "identifier": "tslib-raw-hourly-leap-boundary-v1",
            "description": "CPU float32 batch=1/2 cases crossing leap-day and month boundaries with raw encoder/decoder calendar marks.",
        },
        "tolerances": detail["tolerances"],
        "modes": ["eval", "train"],
        "checks": {
            "outputs": _check(passed, evidence, max_abs=errors["outputs"][0], max_rel=errors["outputs"][1]),
            "intermediates": _check(passed, evidence, max_abs=errors["intermediates"][0], max_rel=errors["intermediates"][1]),
            "input_gradients": _check(passed, evidence, max_abs=errors["input_gradients"][0], max_rel=errors["input_gradients"][1]),
            "parameter_gradients": _check(passed, evidence, max_abs=errors["parameter_gradients"][0], max_rel=errors["parameter_gradients"][1]),
            "train_eval": _check(passed, evidence, modes="eval,train"),
            "buffers": _check(passed, evidence, mapped_buffers=first["mapped_buffers"]),
            "serialization": _check(passed, evidence, max_abs=serial_error),
            "preprocessing": _check(passed, evidence, contract="raw six-column calendar marks exactly reproduce pinned hourly timeF features"),
            "boundaries": _check(passed, evidence, cases="batch=1,batch=2; leap-day; month boundary; encoder and decoder marks"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer-checkout", type=Path, required=True)
    parser.add_argument("--etsformer-checkout", type=Path, required=True)
    parser.add_argument("--models", nargs="*", choices=tuple(SOURCES), default=list(SOURCES))
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    checkout_for = {
        "Transformer": args.transformer_checkout.resolve(),
        "Informer": args.transformer_checkout.resolve(),
        "ETSformer": args.etsformer_checkout.resolve(),
    }
    all_passed = True
    for name in args.models:
        upstream_cls = _load_upstream(name, checkout_for[name])
        detail = verify_model(name, upstream_cls)
        all_passed = all_passed and bool(detail["passed"])
        print(f"{name}: {'PASS' if detail['passed'] else 'FAIL'}")
        if args.write:
            path = ROOT / "verification" / "parity" / f"{name}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(detail, indent=2, sort_keys=True) + "\n")
            write_verification_result(
                ROOT / DEFAULT_INDEX,
                canonical_result(name, detail, path),
            )
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
