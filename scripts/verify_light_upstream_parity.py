#!/usr/bin/env python3
"""Generate exact pinned-source parity evidence for CATS and SegRNN.

xPatch is intentionally excluded: its pinned upstream EMA implementation
hard-codes CUDA, so the defining path cannot be executed in the repository's
CPU verification environment.  Running only its ``reg`` ablation would not
qualify the declared upstream implementation.
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
from models.cats.model import Model as LocalCATS  # noqa: E402
from models.segrnn.model import Model as LocalSegRNN  # noqa: E402


SOURCES = {
    "CATS": {
        "url": "https://github.com/dongbeank/CATS",
        "revision": "58854fc759d608ce400f378be83f4513960e505d",
        "license": "MIT",
        "file": "models/CATS.py",
        "sha256": "324acdb3036b2c9c019c5caa2d78eebe95f704ca230e633ab3c707275969441b",
        "license_markers": ("MIT License", "Copyright (c) 2024 Dongbin Kim"),
    },
    "SegRNN": {
        "url": "https://github.com/lss-1138/SegRNN",
        "revision": "8e869ecfdf1daab3a0ba14d1d620796c1a5d2c4f",
        "license": "Apache-2.0",
        "file": "models/SegRNN.py",
        "sha256": "b7dd6ba15600fbb67f5137e58d45795c75a13a1661bc615ffe9c7b145bf1da95",
        "license_markers": ("Apache License", "Version 2.0"),
    },
}


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


def _ignored_optional_contract(
    local: nn.Module, values: torch.Tensor, pred_len: int
) -> dict[str, Any]:
    """Prove that the repository-only optional signature remains inert."""
    batch, seq_len, channels = values.shape
    local.eval()
    with torch.no_grad():
        expected = local(values)
        actual = local(
            values,
            torch.randn(batch, seq_len, 4),
            torch.randn(batch, pred_len, channels),
            torch.randn(batch, pred_len, 4),
            torch.zeros(seq_len, seq_len, dtype=torch.bool),
        )
    difference = (expected - actual).abs()
    return {
        "passed": bool(torch.equal(expected, actual)),
        "max_abs": float(difference.max()) if difference.numel() else 0.0,
        "contract": "marks, decoder inputs, and mask are accepted and ignored",
    }


def _case_payload(
    report, state_map, module_map, serialization, optional_inputs, fixture
):
    passed = (
        report.passed
        and all(item["passed"] for item in serialization.values())
        and optional_inputs["passed"]
    )
    return {
        "passed": passed,
        "fixture": fixture,
        "state_map": state_map,
        "module_map": module_map,
        "serialization": serialization,
        "optional_inputs": optional_inputs,
        "report": report.to_dict(),
    }


def _cats_case(upstream_class, *, seq_len, pred_len, channels, patch_len, stride,
               independence, padding_patch, batch):
    config = SimpleNamespace(
        seq_len=seq_len,
        pred_len=pred_len,
        dec_in=channels,
        d_layers=2,
        n_heads=2,
        d_model=8,
        d_ff=16,
        dropout=0.1,
        query_independence=independence,
        patch_len=patch_len,
        stride=stride,
        padding_patch=padding_patch,
        store_attn=False,
        QAM_start=0.1,
        QAM_end=0.5,
    )
    local = LocalCATS(
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=channels,
        patch_len=patch_len,
        stride=stride,
        n_layers=config.d_layers,
        d_model=config.d_model,
        n_heads=config.n_heads,
        d_ff=config.d_ff,
        dropout=config.dropout,
        attn_dropout=0.0,
        query_independence=independence,
        padding_patch=padding_patch,
        store_attn=False,
        QAM_start=config.QAM_start,
        QAM_end=config.QAM_end,
    )
    upstream = upstream_class(config)
    state_map = {
        name: "model." + name[len("net."):]
        for name in local.state_dict()
    }
    module_map = {
        "net.backbone.W_P": "model.backbone.W_P",
        "net.backbone.decoder.layers.0.cross_attn":
            "model.backbone.decoder.layers.0.cross_attn",
        "net.backbone.decoder.layers.1.ffn":
            "model.backbone.decoder.layers.1.ffn",
        "net.proj": "model.proj",
    }
    inputs = (torch.randn(batch, seq_len, channels),)
    report = compare_model_parity(
        local,
        upstream,
        inputs,
        state_map=state_map,
        module_map=module_map,
        modes=("eval", "train"),
        compare_gradients=True,
        seed=31415,
        atol=1e-6,
        rtol=1e-5,
    )
    _assert_complete(report, local)
    serialization = {
        "local": _round_trip(local, inputs),
        "upstream": _round_trip(upstream, inputs),
    }
    optional_inputs = _ignored_optional_contract(local, inputs[0], pred_len)
    return _case_payload(
        report, state_map, module_map, serialization, optional_inputs, vars(config)
    )


def _segrnn_case(upstream_class, *, seq_len, pred_len, channels, seg_len,
                 d_model, batch):
    config = SimpleNamespace(
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=channels,
        d_model=d_model,
        dropout=0.1,
        rnn_type="gru",
        dec_way="pmf",
        seg_len=seg_len,
        channel_id=True,
        revin=False,
    )
    local = LocalSegRNN(
        seq_len=seq_len,
        pred_len=pred_len,
        enc_in=channels,
        d_model=d_model,
        dropout=config.dropout,
        seg_len=seg_len,
    )
    upstream = upstream_class(config)
    state_map = {}
    for name in local.state_dict():
        suffix = name[len("model."):]
        if suffix.startswith("value_embedding."):
            suffix = "valueEmbedding." + suffix[len("value_embedding."):]
        state_map[name] = suffix
    module_map = {
        "model.value_embedding": "valueEmbedding",
        "model.rnn": "rnn",
        "model.predict": "predict",
    }
    inputs = (torch.randn(batch, seq_len, channels),)
    report = compare_model_parity(
        local,
        upstream,
        inputs,
        state_map=state_map,
        module_map=module_map,
        modes=("eval", "train"),
        compare_gradients=True,
        seed=31415,
        atol=1e-6,
        rtol=1e-5,
    )
    _assert_complete(report, local)
    serialization = {
        "local": _round_trip(local, inputs),
        "upstream": _round_trip(upstream, inputs),
    }
    optional_inputs = _ignored_optional_contract(local, inputs[0], pred_len)
    return _case_payload(
        report, state_map, module_map, serialization, optional_inputs, vars(config)
    )


def verify_model(name: str, upstream_class, *, command: str):
    torch.manual_seed(31415)
    if name == "CATS":
        cases = {
            "shared_queries": _cats_case(
                upstream_class, seq_len=8, pred_len=4, channels=2,
                patch_len=4, stride=4, independence=False,
                padding_patch=None, batch=2,
            ),
            "independent_padded": _cats_case(
                upstream_class, seq_len=6, pred_len=3, channels=3,
                patch_len=3, stride=3, independence=True,
                padding_patch="end", batch=1,
            ),
            "minimum_sequence": _cats_case(
                upstream_class, seq_len=2, pred_len=2, channels=1,
                patch_len=2, stride=2, independence=False,
                padding_patch=None, batch=1,
            ),
        }
    elif name == "SegRNN":
        cases = {
            "multivariate": _segrnn_case(
                upstream_class, seq_len=8, pred_len=4, channels=3,
                seg_len=2, d_model=8, batch=2,
            ),
            "single_channel": _segrnn_case(
                upstream_class, seq_len=6, pred_len=3, channels=1,
                seg_len=3, d_model=4, batch=1,
            ),
            "minimum_sequence": _segrnn_case(
                upstream_class, seq_len=1, pred_len=1, channels=1,
                seg_len=1, d_model=2, batch=1,
            ),
        }
    else:
        raise KeyError(name)
    source = SOURCES[name]
    return {
        "schema_version": 1,
        "model": name,
        "passed": all(case["passed"] for case in cases.values()),
        "source": {
            **{key: source[key] for key in ("url", "revision", "license", "file")},
            "file_sha256": source["sha256"],
        },
        "upstream_execution": "exact-pinned-checkout",
        "mapping_version": "light-upstream-v1",
        "command": command,
        "deterministic": {"seed": 31415, "device": "cpu"},
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
    return max(float(item["max_abs"]) for item in values), max(
        float(item["max_rel"]) for item in values
    )


def _check(passed, evidence, **metrics):
    return {"passed": passed, "evidence": evidence, "metrics": metrics}


def canonical_result(name, detail, detail_path):
    fields = next(item for item in model_records(ROOT) if item["name"] == name)
    relative = detail_path.relative_to(ROOT).as_posix()
    script_relative = "scripts/verify_light_upstream_parity.py"
    test_relative = "tests/test_light_upstream_parity.py"
    evidence = [relative, script_relative, test_relative]
    output_abs, output_rel = _errors(detail, "outputs")
    mid_abs, mid_rel = _errors(detail, "intermediates")
    input_abs, input_rel = _errors(detail, "input_gradients")
    param_abs, param_rel = _errors(detail, "parameter_gradients")
    serialization = all(
        value["passed"]
        for case in detail["cases"].values()
        for value in case["serialization"].values()
    )
    all_pass = bool(detail["passed"])
    mapped_parameters = max(
        len(case["state_map"]) for case in detail["cases"].values()
    )
    source = SOURCES[name]
    return {
        "schema_version": 1,
        "kind": "upstream-parity",
        "implementation": "upstream",
        "model": name,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "subject_sha256": verification_subject_sha256(ROOT, fields),
        "artifacts": {
            relative: hashlib.sha256(detail_path.read_bytes()).hexdigest(),
            script_relative: hashlib.sha256(
                (ROOT / script_relative).read_bytes()
            ).hexdigest(),
            test_relative: hashlib.sha256(
                (ROOT / test_relative).read_bytes()
            ).hexdigest(),
        },
        "commands": [
            detail["command"],
            "uv run python -m unittest tests.test_light_upstream_parity -v",
            f"uv run tsf repo doctor --backward --models {name}",
        ],
        "environment": {
            "python": platform.python_version(),
            "framework": f"torch {torch.__version__}",
            "dependencies": {"numpy": np.__version__, "torch": torch.__version__},
            "platform": platform.platform(),
            "device": "cpu",
            "dtype": "float32",
            "deterministic": {
                "seed": 31415,
                "algorithms": torch.are_deterministic_algorithms_enabled(),
                "num_threads": torch.get_num_threads(),
            },
        },
        "passed": all_pass and serialization,
        "source": {key: source[key] for key in ("url", "revision", "license")},
        "mapping": {
            "version": "light-upstream-v1",
            "parameters": mapped_parameters,
            "buffers": 0,
        },
        "fixture": {
            "identifier": "light-upstream-boundaries-v1",
            "description": (
                "Seeded CPU cases cover train/eval, batch/channel variation, "
                "minimum valid sequences, and architecture-specific branches."
            ),
        },
        "tolerances": {"atol": 1e-6, "rtol": 1e-5},
        "modes": ["eval", "train"],
        "checks": {
            "outputs": _check(all_pass, evidence, max_abs=output_abs, max_rel=output_rel),
            "intermediates": _check(all_pass, evidence, max_abs=mid_abs, max_rel=mid_rel),
            "input_gradients": _check(all_pass, evidence, max_abs=input_abs, max_rel=input_rel),
            "parameter_gradients": _check(all_pass, evidence, max_abs=param_abs, max_rel=param_rel),
            "train_eval": _check(all_pass, evidence, modes="eval,train"),
            "buffers": _check(all_pass, evidence, mapped_buffers=0, reason="no persistent buffers"),
            "serialization": _check(serialization, evidence, max_abs=0.0),
            "preprocessing": _check(
                all_pass,
                evidence,
                contract=(
                    "identical BLC floating-point history tensors; the "
                    "repository-only marks, decoder inputs, and mask are "
                    "independently proven inert"
                ),
            ),
            "boundaries": _check(all_pass, evidence, cases=",".join(detail["cases"])),
        },
    }


def _load_exact(name: str, checkout: Path):
    source = SOURCES[name]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=checkout, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if revision != source["revision"]:
        raise ValueError(f"{name} checkout is {revision}, expected {source['revision']}")
    license_text = (checkout / "LICENSE").read_text(encoding="utf-8")
    if any(marker not in license_text for marker in source["license_markers"]):
        raise ValueError(f"{name} checkout does not contain its recorded license")
    path = checkout / source["file"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != source["sha256"]:
        raise ValueError(f"{name} source digest is {digest}, expected {source['sha256']}")

    saved = {
        key: value for key, value in sys.modules.items()
        if key == "models" or key.startswith("models.") or
        key == "layers" or key.startswith("layers.")
    }
    for key in saved:
        del sys.modules[key]
    sys.path.insert(0, str(checkout))
    try:
        for package_name in ("models", "layers"):
            package_path = checkout / package_name
            if package_path.is_dir():
                package = ModuleType(package_name)
                package.__path__ = [str(package_path)]  # type: ignore[attr-defined]
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
            if (key == "models" or key.startswith("models.") or
                    key == "layers" or key.startswith("layers.")):
                del sys.modules[key]
        sys.modules.update(saved)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cats-checkout", type=Path, required=True)
    parser.add_argument("--segrnn-checkout", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "verification" / "parity"
    )
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkouts = {"CATS": args.cats_checkout, "SegRNN": args.segrnn_checkout}
    command = (
        "uv run python scripts/verify_light_upstream_parity.py "
        "--cats-checkout <CATS@58854fc759d608ce400f378be83f4513960e505d> "
        "--segrnn-checkout <SegRNN@8e869ecfdf1daab3a0ba14d1d620796c1a5d2c4f>"
    )
    passed = True
    for name, checkout in checkouts.items():
        detail = verify_model(
            name, _load_exact(name, checkout.resolve()), command=command
        )
        output = args.output_dir / f"{name}.json"
        output.write_text(json.dumps(detail, indent=2) + "\n", encoding="utf-8")
        write_verification_result(
            ROOT / DEFAULT_INDEX, canonical_result(name, detail, output)
        )
        print(f"{'PASS' if detail['passed'] else 'FAIL'} {name}: {output}")
        passed &= detail["passed"]
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
