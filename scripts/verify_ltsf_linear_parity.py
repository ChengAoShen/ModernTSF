#!/usr/bin/env python3
"""Generate checked parity evidence for the pinned LTSF-Linear family."""

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
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark.parity import compare_model_parity  # noqa: E402
from benchmark.catalog_metadata import model_records  # noqa: E402
from benchmark.verification_results import (  # noqa: E402
    DEFAULT_INDEX,
    verification_subject_sha256,
    write_verification_result,
)
from models.dlinear.model import Model as LocalDLinear  # noqa: E402
from models.linear.model import Model as LocalLinear  # noqa: E402
from models.nlinear.model import Model as LocalNLinear  # noqa: E402
from verification.fixtures.ltsf_linear import (  # noqa: E402
    Config,
    DLinear as UpstreamDLinear,
    Linear as UpstreamLinear,
    NLinear as UpstreamNLinear,
    SOURCE_LICENSE,
    SOURCE_REVISION,
    SOURCE_URL,
)


MODELS = {
    "Linear": (LocalLinear, UpstreamLinear),
    "NLinear": (LocalNLinear, UpstreamNLinear),
    "DLinear": (LocalDLinear, UpstreamDLinear),
}

UPSTREAM_FILE_SHA256 = {
    "Linear": "1c583fb30ecbeb82e27926d886088543b6cd3b0b911c56bd0a5ff97d78f53142",
    "NLinear": "cabdcf5ac2658faa45b8887846182d222eb0a98c42059079c533df0fe689d879",
    "DLinear": "0893b53cb6473d6bdca7aeca514cb3ee12efa6df227c29c4469571c9711451cc",
}


def _state_map(name: str, individual: bool, channels: int) -> dict[str, str]:
    if name in {"Linear", "NLinear"}:
        local_prefix = "model.projection"
        upstream_prefix = "Linear"
        branches = ((local_prefix, upstream_prefix),)
    else:
        branches = (
            ("seasonal_projection", "Linear_Seasonal"),
            ("trend_projection", "Linear_Trend"),
        )
    mapping: dict[str, str] = {}
    for local_prefix, upstream_prefix in branches:
        if individual:
            for index in range(channels):
                for suffix in ("weight", "bias"):
                    mapping[f"{local_prefix}.linears.{index}.{suffix}"] = (
                        f"{upstream_prefix}.{index}.{suffix}"
                    )
        else:
            for suffix in ("weight", "bias"):
                mapping[f"{local_prefix}.linear.{suffix}"] = (
                    f"{upstream_prefix}.{suffix}"
                )
    return mapping


def _module_map(name: str, individual: bool, channels: int) -> dict[str, str]:
    if name in {"Linear", "NLinear"}:
        if individual:
            return {
                f"model.projection.linears.{index}": f"Linear.{index}"
                for index in range(channels)
            }
        return {"model.projection.linear": "Linear"}
    mapping = {"decomposition": "decompsition"}
    if individual:
        for branch, upstream_branch in (
            ("seasonal_projection", "Linear_Seasonal"),
            ("trend_projection", "Linear_Trend"),
        ):
            mapping.update(
                {
                    f"{branch}.linears.{index}": f"{upstream_branch}.{index}"
                    for index in range(channels)
                }
            )
    else:
        mapping.update(
            {
                "seasonal_projection.linear": "Linear_Seasonal",
                "trend_projection.linear": "Linear_Trend",
            }
        )
    return mapping


def _round_trip(model: torch.nn.Module, values: torch.Tensor) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        expected = model(values)
    buffer = BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = deepcopy(model)
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    with torch.no_grad():
        actual = restored(values)
    difference = (expected - actual).abs()
    return {
        "passed": bool(torch.equal(expected, actual)),
        "max_abs": float(difference.max()) if difference.numel() else 0.0,
    }


def verify_case(
    name: str,
    individual: bool,
    *,
    upstream_class: type[torch.nn.Module] | None = None,
    batch_size: int | None = None,
    seq_len: int = 25,
    pred_len: int = 7,
    channels: int | None = None,
) -> dict[str, Any]:
    batch_size = batch_size if batch_size is not None else (2 if individual else 1)
    channels = channels if channels is not None else (3 if individual else 1)
    seed = 1729
    torch.manual_seed(seed)
    config = Config(seq_len, pred_len, channels, individual)
    local_class, fixture_class = MODELS[name]
    upstream_class = upstream_class or fixture_class
    local_kwargs = {
        "c_in": channels,
        "seq_len": seq_len,
        "pred_len": pred_len,
        "individual": individual,
    }
    if name == "DLinear":
        local_kwargs["kernel_size"] = 25
    local = local_class(**local_kwargs)
    upstream = upstream_class(config)
    values = torch.randn(batch_size, seq_len, channels, dtype=torch.float32)
    state_map = _state_map(name, individual, channels)
    report = compare_model_parity(
        local,
        upstream,
        (values,),
        state_map=state_map,
        module_map=_module_map(name, individual, channels),
        modes=("eval", "train"),
        compare_gradients=True,
        seed=seed,
        atol=1e-6,
        rtol=1e-5,
    )
    expected_parameter_grads = len(dict(local.named_parameters()))
    for mode_name, mode in report.modes.items():
        if len(mode.parameter_gradients) != expected_parameter_grads:
            raise AssertionError(
                f"{name}/{individual}/{mode_name} omitted parameter gradients"
            )
        if len(mode.input_gradients) != 1:
            raise AssertionError(f"{name}/{individual}/{mode_name} omitted input gradient")
    serialization = {
        "local": _round_trip(local, values),
        "upstream": _round_trip(upstream, values),
    }
    passed = report.passed and all(item["passed"] for item in serialization.values())
    return {
        "passed": passed,
        "fixture": {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "pred_len": pred_len,
            "channels": channels,
            "individual": individual,
            "dtype": "float32",
        },
        "state_map": state_map,
        "module_map": _module_map(name, individual, channels),
        "serialization": serialization,
        "report": report.to_dict(),
    }


def verify_model(
    name: str,
    *,
    upstream_class: type[torch.nn.Module] | None = None,
    upstream_execution: str = "checked-thin-fixture",
    command: str = "uv run python scripts/verify_ltsf_linear_parity.py",
) -> dict[str, Any]:
    cases = {
        "shared": verify_case(name, False, upstream_class=upstream_class),
        "individual": verify_case(name, True, upstream_class=upstream_class),
        "minimum_sequence": verify_case(
            name,
            False,
            upstream_class=upstream_class,
            batch_size=1,
            seq_len=1,
            pred_len=1,
            channels=1,
        ),
    }
    return {
        "schema_version": 1,
        "model": name,
        "passed": all(case["passed"] for case in cases.values()),
        "source": {
            "url": SOURCE_URL,
            "revision": SOURCE_REVISION,
            "license": SOURCE_LICENSE,
            "files": [f"models/{name}.py"],
            "file_sha256": UPSTREAM_FILE_SHA256[name],
        },
        "upstream_execution": upstream_execution,
        "mapping_version": "ltsf-linear-v1",
        "command": command,
        "deterministic": {"seed": 1729, "device": "cpu"},
        "tolerances": {"atol": 1e-6, "rtol": 1e-5},
        "cases": cases,
    }


def _errors(record: dict[str, Any], group: str) -> tuple[float, float]:
    comparisons = [
        comparison
        for case in record["cases"].values()
        for mode in case["report"]["modes"].values()
        for comparison in mode[group].values()
    ]
    return (
        max(float(item["max_abs"]) for item in comparisons),
        max(float(item["max_rel"]) for item in comparisons),
    )


def _check(
    passed: bool,
    evidence: list[str],
    **metrics: float | int | str,
) -> dict[str, Any]:
    return {"passed": passed, "evidence": evidence, "metrics": metrics}


def canonical_result(name: str, detail: dict[str, Any], detail_path: Path) -> dict[str, Any]:
    fields = next(item for item in model_records(ROOT) if item["name"] == name)
    relative_detail = detail_path.relative_to(ROOT).as_posix()
    all_cases_pass = bool(detail["passed"])
    output_abs, output_rel = _errors(detail, "outputs")
    intermediate_abs, intermediate_rel = _errors(detail, "intermediates")
    input_abs, input_rel = _errors(detail, "input_gradients")
    parameter_abs, parameter_rel = _errors(detail, "parameter_gradients")
    parameter_count = len(
        detail["cases"]["individual"]["state_map"]
    )
    serialization_passed = all(
        item["passed"]
        for case in detail["cases"].values()
        for item in case["serialization"].values()
    )
    evidence = [relative_detail, "tests/test_ltsf_linear_parity.py"]
    return {
        "schema_version": 1,
        "kind": "upstream-parity",
        "implementation": "upstream",
        "model": name,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "subject_sha256": verification_subject_sha256(ROOT, fields),
        "artifacts": {
            relative_detail: hashlib.sha256(detail_path.read_bytes()).hexdigest(),
        },
        "commands": [
            str(detail["command"]),
            "uv run python -m unittest tests.test_ltsf_linear_parity -v",
            f"uv run tsf repo doctor --backward --models {name}",
        ],
        "environment": {
            "python": platform.python_version(),
            "framework": f"torch {torch.__version__}",
            "dependencies": {
                "numpy": np.__version__,
                "torch": torch.__version__,
            },
            "platform": platform.platform(),
            "device": "cpu",
            "dtype": "float32",
            "deterministic": {
                "seed": 1729,
                "algorithms": torch.are_deterministic_algorithms_enabled(),
                "num_threads": torch.get_num_threads(),
            },
        },
        "passed": all_cases_pass and serialization_passed,
        "source": {
            "url": SOURCE_URL,
            "revision": SOURCE_REVISION,
            "license": SOURCE_LICENSE,
        },
        "mapping": {
            "version": "ltsf-linear-v1",
            "parameters": parameter_count,
            "buffers": 0,
        },
        "fixture": {
            "identifier": "ltsf-linear-shared-individual-minimum-v1",
            "description": (
                "Seeded float32 CPU inputs cover shared batch/channel=1, "
                "individual batch=2/channel=3, and the minimum seq_len=pred_len=1."
            ),
        },
        "tolerances": {"atol": 1e-6, "rtol": 1e-5},
        "modes": ["eval", "train"],
        "checks": {
            "outputs": _check(
                all_cases_pass,
                evidence,
                max_abs=output_abs,
                max_rel=output_rel,
            ),
            "intermediates": _check(
                all_cases_pass,
                evidence,
                max_abs=intermediate_abs,
                max_rel=intermediate_rel,
            ),
            "input_gradients": _check(
                all_cases_pass,
                evidence,
                max_abs=input_abs,
                max_rel=input_rel,
            ),
            "parameter_gradients": _check(
                all_cases_pass,
                evidence,
                max_abs=parameter_abs,
                max_rel=parameter_rel,
            ),
            "train_eval": _check(
                all_cases_pass,
                evidence,
                modes="eval,train",
            ),
            "buffers": _check(
                all_cases_pass,
                evidence,
                mapped_buffers=0,
                reason="both implementations have no persistent buffers",
            ),
            "serialization": _check(
                serialization_passed,
                evidence,
                max_abs=0.0,
            ),
            "preprocessing": _check(
                all_cases_pass,
                evidence,
                contract="identical raw BLC tensor; no external transform",
            ),
            "boundaries": _check(
                all_cases_pass,
                evidence,
                cases="shared,individual,minimum_sequence",
            ),
        },
    }


def _load_exact_upstream(checkout: Path) -> dict[str, type[torch.nn.Module]]:
    import subprocess

    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != SOURCE_REVISION:
        raise ValueError(
            f"upstream checkout is {revision}, expected pinned {SOURCE_REVISION}"
        )
    classes: dict[str, type[torch.nn.Module]] = {}
    for name, expected_sha256 in UPSTREAM_FILE_SHA256.items():
        path = checkout / "models" / f"{name}.py"
        actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"upstream {name}.py digest is {actual_sha256}, expected {expected_sha256}"
            )
        spec = importlib.util.spec_from_file_location(f"ltsf_linear_{name}", path)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot load upstream source file: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        classes[name] = module.Model
    return classes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "verification" / "parity",
    )
    parser.add_argument(
        "--upstream-checkout",
        type=Path,
        required=True,
        help="LTSF-Linear checkout exactly at the recorded revision",
    )
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    upstream_classes = _load_exact_upstream(args.upstream_checkout.resolve())
    execution = "exact-pinned-checkout"
    command = (
        "uv run python scripts/verify_ltsf_linear_parity.py "
        "--upstream-checkout <LTSF-Linear@0c113668a3b88c4c4ee586b8c5ec3e539c4de5a6>"
    )
    passed = True
    for name in MODELS:
        record = verify_model(
            name,
            upstream_class=upstream_classes.get(name),
            upstream_execution=execution,
            command=command,
        )
        output = args.output_dir / f"{name}.json"
        output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        write_verification_result(
            ROOT / DEFAULT_INDEX,
            canonical_result(name, record, output),
        )
        print(f"{'PASS' if record['passed'] else 'FAIL'} {name}: {output}")
        passed &= bool(record["passed"])
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
