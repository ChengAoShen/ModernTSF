#!/usr/bin/env python3
"""Generate strict parity evidence for FilterNet models and TimeKAN.

The script deliberately requires exact local checkouts of both upstream
repositories.  It validates their revisions and the hashes of the executed
source files before importing the upstream classes.
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
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


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
from models.paifilter.model import Model as LocalPaiFilter  # noqa: E402
from models.texfilter.model import Model as LocalTexFilter  # noqa: E402
from models.timekan.model import Model as LocalTimeKAN  # noqa: E402


FILTER_URL = "https://github.com/aikunyi/FilterNet"
FILTER_REVISION = "cdb321c4e338e0c07b45cee92f54b3c5bd5a809e"
TIMEKAN_URL = "https://github.com/huangst21/TimeKAN"
TIMEKAN_REVISION = "3a7c366a9e8547fd8840c5d27f25ee3e30615e33"
SOURCE_LICENSE = "Apache-2.0"
SOURCE_HASHES = {
    "FilterNet": {
        "LICENSE": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
        "models/PaiFilter.py": "9e4d25da30e5607402994b7969bb3ab89cf364c7d61e5b4f9e0c47f40cf5a570",
        "models/TexFilter.py": "407ee405aae05478ce77d85251c7516273424197c0de300b43d30ed6220e8ba2",
        "layers/RevIN.py": "1e08d0b3679f4ef79ac7f98a7f719f8d7f8a04595022ed34f702c7d70c0c4d90",
    },
    "TimeKAN": {
        "LICENSE": "43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1",
        "models/TimeKAN.py": "0876106c4ddffc6a9158a777706c7ef7bf36a4026325464cf200f5901d8a936f",
        "layers/ChebyKANLayer.py": "82451a6f9e7386516cbfb71d72e3bc2084a3a99d7ef08f1b1f9f0a3643331d69",
        "layers/StandardNorm.py": "3f690dc5fc0e395d4f1ffe6396717548d18bd04f4518258308bbe5f392703f5f",
        "layers/Autoformer_EncDec.py": "48745b4bb647355e9845792a855df9c59fd7df7fcc664c765351fec390c4073e",
        "layers/Embed.py": "ba3a4db09f0a2a0187468e8ee03fe4a8514c9605823bb8afaa08d105b682018b",
    },
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_checkout(
    checkout: Path,
    revision: str,
    hashes: dict[str, str],
) -> None:
    actual_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if actual_revision != revision:
        raise ValueError(f"upstream checkout is {actual_revision}, expected {revision}")
    for relative, expected in hashes.items():
        actual = _sha256(checkout / relative)
        if actual != expected:
            raise ValueError(
                f"upstream {relative} digest is {actual}, expected {expected}"
            )


def _clear_upstream_layers() -> None:
    for name in tuple(sys.modules):
        if name == "layers" or name.startswith("layers."):
            del sys.modules[name]


def _load_class(checkout: Path, relative: str, module_name: str) -> type[torch.nn.Module]:
    _clear_upstream_layers()
    sys.path.insert(0, str(checkout))
    try:
        path = checkout / relative
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ValueError(f"cannot load exact upstream source: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Model
    finally:
        sys.path.remove(str(checkout))


def _round_trip(model: torch.nn.Module, inputs: tuple[Any, ...]) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        expected = model(*inputs)
    buffer = BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = deepcopy(model)
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    restored.eval()
    with torch.no_grad():
        actual = restored(*inputs)
    difference = (expected - actual).abs()
    return {
        "passed": bool(torch.equal(expected, actual)),
        "max_abs": float(difference.max()) if difference.numel() else 0.0,
    }


def _active_parameter_names(model: torch.nn.Module) -> set[str]:
    return {name for name, parameter in model.named_parameters() if parameter.grad is not None}


def _run_case(
    *,
    name: str,
    local: torch.nn.Module,
    upstream: torch.nn.Module,
    values: torch.Tensor,
    state_map: dict[str, str],
    module_map: dict[str, str],
    seed: int,
) -> dict[str, Any]:
    inputs = (values, None, None, None)
    report = compare_model_parity(
        local,
        upstream,
        inputs,
        state_map=state_map,
        module_map=module_map,
        modes=("eval", "train"),
        compare_gradients=True,
        seed=seed,
        atol=1e-6,
        rtol=1e-5,
    )
    compared = {
        key
        for mode in report.modes.values()
        for key in mode.parameter_gradients
    }
    active_local = _active_parameter_names(local)
    active_upstream = _active_parameter_names(upstream)
    mapped_active_upstream = {state_map[name] for name in active_local}
    omitted_local = sorted(active_local - compared)
    omitted_upstream = sorted(active_upstream - mapped_active_upstream)
    active_complete = not omitted_local and not omitted_upstream
    serialization = {
        "local": _round_trip(local, inputs),
        "upstream": _round_trip(upstream, inputs),
    }
    return {
        "passed": bool(
            report.passed
            and active_complete
            and all(item["passed"] for item in serialization.values())
        ),
        "fixture": {
            "batch_size": values.shape[0],
            "seq_len": values.shape[1],
            "channels": values.shape[2],
            "dtype": str(values.dtype).removeprefix("torch."),
        },
        "state_map": state_map,
        "module_map": module_map,
        "active_parameters": {
            "local": sorted(active_local),
            "upstream": sorted(active_upstream),
            "omitted_local": omitted_local,
            "omitted_upstream": omitted_upstream,
        },
        "serialization": serialization,
        "report": report.to_dict(),
    }


def _filter_cases(
    name: str,
    local_class: type[torch.nn.Module],
    upstream_class: type[torch.nn.Module],
) -> dict[str, dict[str, Any]]:
    definitions = {
        "primary": (2, 12, 5, 3),
        "minimum": (1, 1, 1, 1),
        "alternate_length": (1, 13, 4, 2),
    }
    cases = {}
    for index, (case_name, (batch, seq_len, pred_len, channels)) in enumerate(
        definitions.items()
    ):
        seed = 4100 + index
        common = {
            "seq_len": seq_len,
            "pred_len": pred_len,
            "enc_in": channels,
            "hidden_size": 7,
        }
        if name == "TexFilter":
            common.update(embed_size=8, dropout=0.2)
        torch.manual_seed(seed)
        local = local_class(**common)
        upstream = upstream_class(SimpleNamespace(**common))
        state_map = {
            local_name: local_name.removeprefix("model.")
            for local_name in local.state_dict()
        }
        module_map = {
            "model.revin_layer": "revin_layer",
            "model.fc": "fc",
        }
        if name == "TexFilter":
            module_map.update(
                {
                    "model.embedding": "embedding",
                    "model.layernorm": "layernorm",
                    "model.layernorm1": "layernorm1",
                    "model.dropout": "dropout",
                    "model.output": "output",
                }
            )
        values = torch.randn(batch, seq_len, channels)
        cases[case_name] = _run_case(
            name=name,
            local=local,
            upstream=upstream,
            values=values,
            state_map=state_map,
            module_map=module_map,
            seed=seed,
        )
    return cases


def _timekan_cases(upstream_class: type[torch.nn.Module]) -> dict[str, dict[str, Any]]:
    definitions = {
        "primary": (2, 16, 4, 3, 2),
        "minimum_downsample": (1, 2, 1, 1, 2),
        "alternate_length": (1, 12, 3, 2, 3),
    }
    cases = {}
    for index, (case_name, (batch, seq_len, pred_len, channels, window)) in enumerate(
        definitions.items()
    ):
        seed = 5100 + index
        kwargs = {
            "seq_len": seq_len,
            "pred_len": pred_len,
            "enc_in": channels,
            "c_out": channels,
            "d_model": 4,
            "e_layers": 1,
            "down_sampling_window": window,
            "down_sampling_layers": 1,
            "begin_order": 1,
            "moving_avg": 3,
            "dropout": 0.2,
            "embed": "timeF",
            "freq": "h",
            "use_norm": 1,
        }
        config = SimpleNamespace(
            **kwargs,
            label_len=0,
            task_name="long_term_forecast",
            channel_independence=1,
            use_future_temporal_feature=0,
        )
        torch.manual_seed(seed)
        local = LocalTimeKAN(**kwargs)
        upstream = upstream_class(config)
        state_map = {name: name for name in local.state_dict()}
        upstream_extra = sorted(set(upstream.state_dict()) - set(state_map.values()))
        if upstream_extra != ["enc_embedding.temporal_embedding.embed.weight"]:
            raise AssertionError(f"unexpected TimeKAN upstream-only state: {upstream_extra}")
        values = torch.randn(batch, seq_len, channels)
        case = _run_case(
            name="TimeKAN",
            local=local,
            upstream=upstream,
            values=values,
            state_map=state_map,
            module_map={
                "enc_embedding.value_embedding": "enc_embedding.value_embedding",
                "add_blocks.0.front_block": "add_blocks.0.front_block",
                "add_blocks.0.front_blocks.0": "add_blocks.0.front_blocks.0",
                "predict_layer": "predict_layer",
                "projection_layer": "projection_layer",
                "normalize_layers.0": "normalize_layers.0",
            },
            seed=seed,
        )
        case["upstream_inactive_state"] = upstream_extra
        cases[case_name] = case
    return cases


def _errors(detail: dict[str, Any], group: str) -> tuple[float, float]:
    comparisons = [
        item
        for case in detail["cases"].values()
        for mode in case["report"]["modes"].values()
        for item in mode[group].values()
    ]
    return (
        max(float(item["max_abs"]) for item in comparisons),
        max(float(item["max_rel"]) for item in comparisons),
    )


def _check(passed: bool, evidence: list[str], **metrics: float | int | str):
    return {"passed": passed, "evidence": evidence, "metrics": metrics}


def _canonical_result(name: str, detail: dict[str, Any], detail_path: Path):
    fields = next(item for item in model_records(ROOT) if item["name"] == name)
    relative = detail_path.relative_to(ROOT).as_posix()
    evidence = [relative, "scripts/verify_filter_timekan_parity.py"]
    output_abs, output_rel = _errors(detail, "outputs")
    intermediate_abs, intermediate_rel = _errors(detail, "intermediates")
    input_abs, input_rel = _errors(detail, "input_gradients")
    parameter_abs, parameter_rel = _errors(detail, "parameter_gradients")
    all_passed = bool(detail["passed"])
    serialization_passed = all(
        result["passed"]
        for case in detail["cases"].values()
        for result in case["serialization"].values()
    )
    mapped_buffers = sum(
        1
        for key in detail["cases"]["primary"]["state_map"]
        if key not in dict(
            (LocalTimeKAN if name == "TimeKAN" else
             LocalPaiFilter if name == "PaiFilter" else LocalTexFilter)(
                **detail["local_primary_kwargs"]
            ).named_parameters()
        )
    )
    return {
        "schema_version": 1,
        "kind": "upstream-parity",
        "implementation": "upstream",
        "model": name,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "subject_sha256": verification_subject_sha256(ROOT, fields),
        "artifacts": {relative: _sha256(detail_path)},
        "commands": [
            detail["command"],
            "uv run python -m unittest tests.test_filter_timekan_parity -v",
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
                "seed": 4100 if name != "TimeKAN" else 5100,
                "algorithms": torch.are_deterministic_algorithms_enabled(),
                "num_threads": torch.get_num_threads(),
            },
        },
        "passed": all_passed and serialization_passed,
        "source": detail["source"],
        "mapping": {
            "version": detail["mapping_version"],
            "parameters": len(
                dict(
                    (LocalTimeKAN if name == "TimeKAN" else
                     LocalPaiFilter if name == "PaiFilter" else LocalTexFilter)(
                        **detail["local_primary_kwargs"]
                    ).named_parameters()
                )
            ),
            "buffers": mapped_buffers,
        },
        "fixture": {
            "identifier": detail["fixture_identifier"],
            "description": detail["fixture_description"],
        },
        "tolerances": {"atol": 1e-6, "rtol": 1e-5},
        "modes": ["eval", "train"],
        "checks": {
            "outputs": _check(all_passed, evidence, max_abs=output_abs, max_rel=output_rel),
            "intermediates": _check(all_passed, evidence, max_abs=intermediate_abs, max_rel=intermediate_rel),
            "input_gradients": _check(all_passed, evidence, max_abs=input_abs, max_rel=input_rel),
            "parameter_gradients": _check(all_passed, evidence, max_abs=parameter_abs, max_rel=parameter_rel),
            "train_eval": _check(all_passed, evidence, modes="eval,train"),
            "buffers": _check(all_passed, evidence, mapped_buffers=mapped_buffers),
            "serialization": _check(serialization_passed, evidence, max_abs=0.0),
            "preprocessing": _check(all_passed, evidence, contract="identical raw BLC tensor; marks are None; no external transform"),
            "boundaries": _check(all_passed, evidence, cases=",".join(detail["cases"])),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--filternet-checkout", type=Path, required=True)
    parser.add_argument("--timekan-checkout", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "verification" / "parity"
    )
    args = parser.parse_args()
    filter_checkout = args.filternet_checkout.resolve()
    timekan_checkout = args.timekan_checkout.resolve()
    _validate_checkout(filter_checkout, FILTER_REVISION, SOURCE_HASHES["FilterNet"])
    _validate_checkout(timekan_checkout, TIMEKAN_REVISION, SOURCE_HASHES["TimeKAN"])
    upstream_pai = _load_class(filter_checkout, "models/PaiFilter.py", "exact_pai")
    upstream_tex = _load_class(filter_checkout, "models/TexFilter.py", "exact_tex")
    upstream_timekan = _load_class(timekan_checkout, "models/TimeKAN.py", "exact_timekan")

    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    command = (
        "uv run python scripts/verify_filter_timekan_parity.py "
        "--filternet-checkout <FilterNet@cdb321c4> "
        "--timekan-checkout <TimeKAN@3a7c366a>"
    )
    records = {
        "PaiFilter": {
            "source": {"url": FILTER_URL, "revision": FILTER_REVISION, "license": SOURCE_LICENSE},
            "mapping_version": "filternet-paifilter-v1",
            "local_primary_kwargs": {"seq_len": 12, "pred_len": 5, "enc_in": 3, "hidden_size": 7},
            "fixture_identifier": "filternet-paifilter-boundaries-v1",
            "fixture_description": "Seeded float32 CPU cases cover batch/channel boundaries and three configured sequence lengths, with marks omitted by contract.",
            "cases": _filter_cases("PaiFilter", LocalPaiFilter, upstream_pai),
        },
        "TexFilter": {
            "source": {"url": FILTER_URL, "revision": FILTER_REVISION, "license": SOURCE_LICENSE},
            "mapping_version": "filternet-texfilter-v1",
            "local_primary_kwargs": {"seq_len": 12, "pred_len": 5, "enc_in": 3, "embed_size": 8, "hidden_size": 7, "dropout": 0.2},
            "fixture_identifier": "filternet-texfilter-boundaries-v1",
            "fixture_description": "Seeded float32 CPU cases cover train/eval dropout, batch/channel boundaries and three configured sequence lengths.",
            "cases": _filter_cases("TexFilter", LocalTexFilter, upstream_tex),
        },
        "TimeKAN": {
            "source": {"url": TIMEKAN_URL, "revision": TIMEKAN_REVISION, "license": SOURCE_LICENSE},
            "mapping_version": "timekan-forecast-v1",
            "local_primary_kwargs": {"seq_len": 16, "pred_len": 4, "enc_in": 3, "c_out": 3, "d_model": 4, "e_layers": 1, "down_sampling_window": 2, "down_sampling_layers": 1, "begin_order": 1, "moving_avg": 3, "dropout": 0.2, "embed": "timeF", "freq": "h", "use_norm": 1},
            "fixture_identifier": "timekan-forecast-boundaries-v1",
            "fixture_description": "Seeded float32 CPU cases cover batch/channel boundaries, minimum valid downsampling, two window sizes and three configured sequence lengths.",
            "cases": _timekan_cases(upstream_timekan),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    passed = True
    for name, detail in records.items():
        detail.update(
            {
                "schema_version": 1,
                "model": name,
                "upstream_execution": "exact-pinned-checkout",
                "source_files": SOURCE_HASHES["TimeKAN" if name == "TimeKAN" else "FilterNet"],
                "tolerances": {"atol": 1e-6, "rtol": 1e-5},
                "command": command,
            }
        )
        detail["passed"] = all(case["passed"] for case in detail["cases"].values())
        output = args.output_dir / f"{name}.json"
        output.write_text(json.dumps(detail, indent=2) + "\n", encoding="utf-8")
        write_verification_result(
            ROOT / DEFAULT_INDEX,
            _canonical_result(name, detail, output),
        )
        print(f"{'PASS' if detail['passed'] else 'FAIL'} {name}: {output}")
        passed &= bool(detail["passed"])
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
