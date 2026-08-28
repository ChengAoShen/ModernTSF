#!/usr/bin/env python3
"""Generate strict parity evidence for pinned N-BEATS and N-HiTS sources."""

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
from types import ModuleType
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
from models.nbeats.model import Model as LocalNBeats  # noqa: E402
from models.nhits.model import Model as LocalNHiTS  # noqa: E402


SOURCES = {
    "NBeats": {
        "url": "https://github.com/philipperemy/n-beats",
        "revision": "06a4e209ada80bf1f403ced5228261784dfb26ed",
        "license": "MIT",
        "file": "nbeats_pytorch/model.py",
        "sha256": "e35f805f837e1cc3dfd5340f59c7ecc310dca62e4673d7299ddcca5323d03d8c",
        "license_file": "LICENSE",
        "license_sha256": "66cd5a118f5f002110b6783a89eed5e3223e598c1cdb76994188de3bc0e67d6f",
    },
    "NHiTS": {
        "url": "https://github.com/Nixtla/neuralforecast",
        "revision": "6c4f3e557d0ed672314323edba972eb550cb3550",
        "license": "Apache-2.0",
        "file": "neuralforecast/models/nhits.py",
        "sha256": "5d97000d85806343a048c7b75a4f68129656bbece57cd603fcefaae74e045fc8",
        "license_file": "LICENSE",
        "license_sha256": "d392f3d2969e354d7e6aef8fc6df5590ef5c20eea4b946191bd76a6138b4e81c",
    },
}

SEED = 314159
ATOL = 1e-6
RTOL = 1e-5


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _checked_source(checkout: Path, model: str) -> Path:
    source = SOURCES[model]
    revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != source["revision"]:
        raise ValueError(
            f"{model} checkout revision mismatch: {revision} != {source['revision']}"
        )
    license_path = checkout / source["license_file"]
    if not license_path.is_file() or _sha256(license_path) != source["license_sha256"]:
        raise ValueError(f"{model} pinned license file is missing or differs")
    path = checkout / source["file"]
    if not path.is_file():
        raise ValueError(f"{model} source file is missing: {path}")
    actual = _sha256(path)
    if actual != source["sha256"]:
        raise ValueError(
            f"{model} source SHA-256 mismatch: {actual} != {source['sha256']}"
        )
    return path


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_nbeats(checkout: Path) -> ModuleType:
    """Execute the exact pinned upstream N-BEATS module."""
    return _load_module("_moderntsf_pinned_nbeats", _checked_source(checkout, "NBeats"))


def load_nhits(checkout: Path) -> ModuleType:
    """Execute pinned N-HiTS with a minimal BaseModel dependency boundary.

    The architecture file is run unchanged.  The stubs replace only NeuralForecast's
    trainer/data base class and MAE metadata; neither participates in ``NHITS.forward``.
    """
    package = "_moderntsf_pinned_neuralforecast"
    for name in (
        package,
        f"{package}.models",
        f"{package}.common",
        f"{package}.losses",
    ):
        module = ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = module

    base_module = ModuleType(f"{package}.common._base_model")

    class BaseModel(nn.Module):
        def __init__(
            self,
            *,
            h,
            input_size,
            futr_exog_list=None,
            hist_exog_list=None,
            stat_exog_list=None,
            loss,
            random_seed=1,
            **_kwargs,
        ):
            super().__init__()
            if any((futr_exog_list, hist_exog_list, stat_exog_list)):
                raise ValueError("the forecast-only parity adapter forbids exogenous inputs")
            self.h = h
            self.input_size = input_size
            self.futr_exog_size = 0
            self.hist_exog_size = 0
            self.stat_exog_size = 0
            self.loss = loss
            self.decompose_forecast = False
            torch.manual_seed(random_seed)

    base_module.BaseModel = BaseModel
    sys.modules[base_module.__name__] = base_module

    loss_module = ModuleType(f"{package}.losses.pytorch")

    class MAE:
        outputsize_multiplier = 1

    loss_module.MAE = MAE
    sys.modules[loss_module.__name__] = loss_module
    return _load_module(
        f"{package}.models.nhits", _checked_source(checkout, "NHiTS")
    )


def _nbeats_basis_buffers(upstream_module: ModuleType, blocks: nn.ModuleList) -> None:
    """Expose immutable upstream-computed bases for buffer parity/state mapping."""
    for stack in blocks:
        for block in stack:
            if isinstance(block, upstream_module.SeasonalityBlock):
                p = block.thetas_dim
                p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
                for name, values in (
                    ("backcast_basis", block.backcast_linspace),
                    ("forecast_basis", block.forecast_linspace),
                ):
                    basis = np.concatenate(
                        (
                            np.array([np.cos(2 * np.pi * i * values) for i in range(p1)]),
                            np.array([np.sin(2 * np.pi * i * values) for i in range(p2)]),
                        ),
                        axis=0,
                    )
                    block.register_buffer(name, torch.tensor(basis, dtype=torch.float32))
            elif isinstance(block, upstream_module.TrendBlock):
                for name, values in (
                    ("backcast_basis", block.backcast_linspace),
                    ("forecast_basis", block.forecast_linspace),
                ):
                    basis = np.array([values**i for i in range(block.thetas_dim)])
                    block.register_buffer(name, torch.tensor(basis, dtype=torch.float32))


class _NBeatsAdapter(nn.Module):
    def __init__(self, module: ModuleType, *, channels: int, **kwargs):
        super().__init__()
        actual = module.NBeatsNet(device=torch.device("cpu"), **kwargs)
        # NBeatsNet stores blocks in Python lists and only registers a ParameterList.
        # Register the same live block objects for named state and intermediate hooks;
        # ``forward`` below still executes the exact upstream NBeatsNet method.
        object.__setattr__(self, "actual", actual)
        self.blocks = nn.ModuleList(nn.ModuleList(stack) for stack in actual.stacks)
        _nbeats_basis_buffers(module, self.blocks)
        self.channels = channels
        self.forecast_length = kwargs["forecast_length"]

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, length, channels = values.shape
        if channels != self.channels:
            raise ValueError("channel boundary differs from the constructed adapter")
        flattened = values.permute(0, 2, 1).reshape(batch * channels, length)
        _backcast, forecast = self.actual(flattened)
        return forecast.reshape(batch, channels, self.forecast_length).permute(0, 2, 1)


class _NHiTSAdapter(nn.Module):
    def __init__(self, actual: nn.Module, *, channels: int, use_norm: bool):
        super().__init__()
        self.upstream = actual
        self.channels = channels
        self.use_norm = use_norm
        self.h = actual.h

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.use_norm:
            means = values.mean(1, keepdim=True).detach()
            centered = values - means
            stdev = torch.sqrt(
                torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            values = centered / stdev
        batch, length, channels = values.shape
        flattened = values.permute(0, 2, 1).reshape(batch * channels, length)
        empty = values.new_empty((batch * channels, 0, 0))
        output = self.upstream(
            {
                "insample_y": flattened.unsqueeze(-1),
                "insample_mask": torch.ones_like(flattened).unsqueeze(-1),
                "futr_exog": empty,
                "hist_exog": empty,
                "stat_exog": values.new_empty((batch * channels, 0)),
            }
        ).squeeze(-1)
        output = output.reshape(batch, channels, self.h).permute(0, 2, 1)
        if self.use_norm:
            output = output * stdev.repeat(1, self.h, 1)
            output = output + means.repeat(1, self.h, 1)
        return output


def _round_trip(model: nn.Module, values: torch.Tensor) -> dict[str, Any]:
    model.eval()
    with torch.no_grad():
        expected = model(values)
    stream = BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = deepcopy(model)
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    with torch.no_grad():
        actual = restored(values)
    difference = (expected - actual).abs()
    return {
        "passed": bool(torch.equal(expected, actual)),
        "max_abs": float(difference.max()) if difference.numel() else 0.0,
    }


def _assert_complete(report, local: nn.Module) -> None:
    active = {name for name, value in local.named_parameters() if value.requires_grad}
    for mode in report.modes.values():
        if set(mode.parameter_gradients) != active:
            missing = sorted(active - set(mode.parameter_gradients))
            raise AssertionError(f"active parameter gradients omitted: {missing}")
        if not mode.input_gradients:
            raise AssertionError("input gradients were omitted")
        if not mode.intermediates:
            raise AssertionError("defining intermediates were omitted")


def _case_payload(report, state_map, module_map, serialization, fixture):
    return {
        "passed": report.passed and all(item["passed"] for item in serialization.values()),
        "fixture": fixture,
        "state_map": state_map,
        "module_map": module_map,
        "serialization": serialization,
        "report": report.to_dict(),
    }


def _nbeats_case(module: ModuleType, **fixture) -> dict[str, Any]:
    kwargs = {
        "stack_types": tuple(fixture["stack_types"]),
        "nb_blocks_per_stack": fixture["nb_blocks_per_stack"],
        "forecast_length": fixture["pred_len"],
        "backcast_length": fixture["seq_len"],
        "thetas_dim": tuple(fixture["thetas_dim"]),
        "share_weights_in_stack": fixture["share_weights_in_stack"],
        "hidden_layer_units": fixture["hidden_layer_units"],
        "nb_harmonics": fixture.get("nb_harmonics"),
    }
    local = LocalNBeats(
        seq_len=fixture["seq_len"],
        pred_len=fixture["pred_len"],
        enc_in=fixture["channels"],
        stack_types=kwargs["stack_types"],
        nb_blocks_per_stack=kwargs["nb_blocks_per_stack"],
        thetas_dim=kwargs["thetas_dim"],
        hidden_layer_units=kwargs["hidden_layer_units"],
        share_weights_in_stack=kwargs["share_weights_in_stack"],
        nb_harmonics=kwargs["nb_harmonics"],
    )
    upstream = _NBeatsAdapter(module, channels=fixture["channels"], **kwargs)
    state_map = {}
    for name in local.state_dict():
        target = name.replace("stacks.", "blocks.", 1)
        # Shared theta layers use opposite canonical alias names upstream.
        stack_index = int(name.split(".")[1])
        if fixture["stack_types"][stack_index] in {"trend", "seasonality"}:
            target = target.replace("theta_b_fc.weight", "theta_f_fc.weight")
        state_map[name] = target
    # ``named_modules`` intentionally de-duplicates shared block objects.  Hook
    # each defining live block once even though state_dict preserves aliases.
    local_modules = dict(local.named_modules())
    upstream_modules = dict(upstream.named_modules())
    module_map = {}
    for stack in range(len(fixture["stack_types"])):
        for block in range(fixture["nb_blocks_per_stack"]):
            local_name = f"stacks.{stack}.{block}"
            upstream_name = f"blocks.{stack}.{block}"
            if local_name in local_modules and upstream_name in upstream_modules:
                module_map[local_name] = upstream_name
    values = torch.randn(
        fixture["batch_size"], fixture["seq_len"], fixture["channels"]
    )
    report = compare_model_parity(
        local,
        upstream,
        (values,),
        state_map=state_map,
        module_map=module_map,
        modes=("eval", "train"),
        compare_gradients=True,
        seed=SEED,
        atol=ATOL,
        rtol=RTOL,
    )
    _assert_complete(report, local)
    serialization = {
        "local": _round_trip(local, values),
        "upstream": _round_trip(upstream, values),
    }
    return _case_payload(report, state_map, module_map, serialization, fixture)


def _nhits_case(module: ModuleType, **fixture) -> dict[str, Any]:
    common = {
        "stack_types": fixture["stack_types"],
        "n_blocks": fixture["n_blocks"],
        "mlp_units": fixture["mlp_units"],
        "n_pool_kernel_size": fixture["n_pool_kernel_size"],
        "n_freq_downsample": fixture["n_freq_downsample"],
        "pooling_mode": fixture["pooling_mode"],
        "interpolation_mode": fixture["interpolation_mode"],
        "activation": fixture["activation"],
    }
    local = LocalNHiTS(
        seq_len=fixture["seq_len"],
        pred_len=fixture["pred_len"],
        enc_in=fixture["channels"],
        dropout=fixture["dropout"],
        use_norm=fixture["use_norm"],
        **common,
    )
    actual = module.NHITS(
        h=fixture["pred_len"],
        input_size=fixture["seq_len"],
        dropout_prob_theta=fixture["dropout"],
        random_seed=SEED,
        scaler_type="identity",
        **common,
    )
    upstream = _NHiTSAdapter(
        actual, channels=fixture["channels"], use_norm=fixture["use_norm"]
    )
    state_map = {name: f"upstream.{name}" for name in local.state_dict()}
    # The upstream basis retains an output-size singleton on each block forecast
    # while the local forecast-only port removes it.  Compare the defining
    # pre-basis theta and pooled-history tensors, whose contracts are identical;
    # final adapted forecasts below cover the intentional singleton removal.
    module_map = {
        **{
            f"blocks.{index}.pooling_layer": f"upstream.blocks.{index}.pooling_layer"
            for index in range(len(local.blocks))
        },
        **{
            f"blocks.{index}.layers": f"upstream.blocks.{index}.layers"
            for index in range(len(local.blocks))
        },
    }
    values = torch.randn(
        fixture["batch_size"], fixture["seq_len"], fixture["channels"]
    )
    report = compare_model_parity(
        local,
        upstream,
        (values,),
        state_map=state_map,
        module_map=module_map,
        modes=("eval", "train"),
        compare_gradients=True,
        seed=SEED,
        atol=ATOL,
        rtol=RTOL,
    )
    _assert_complete(report, local)
    serialization = {
        "local": _round_trip(local, values),
        "upstream": _round_trip(upstream, values),
    }
    return _case_payload(report, state_map, module_map, serialization, fixture)


def verify_model(name: str, module: ModuleType) -> dict[str, Any]:
    if name == "NBeats":
        cases = {
            "mixed_basis": _nbeats_case(
                module,
                batch_size=2,
                seq_len=8,
                pred_len=4,
                channels=2,
                stack_types=["trend", "seasonality", "generic"],
                nb_blocks_per_stack=1,
                thetas_dim=[4, 4, 3],
                hidden_layer_units=7,
                share_weights_in_stack=False,
                nb_harmonics=None,
            ),
            "shared_weights": _nbeats_case(
                module,
                batch_size=1,
                seq_len=5,
                pred_len=3,
                channels=3,
                stack_types=["generic", "trend"],
                nb_blocks_per_stack=2,
                thetas_dim=[3, 3],
                hidden_layer_units=5,
                share_weights_in_stack=True,
                nb_harmonics=None,
            ),
            "minimum": _nbeats_case(
                module,
                batch_size=1,
                seq_len=1,
                pred_len=1,
                channels=1,
                stack_types=["generic"],
                nb_blocks_per_stack=1,
                thetas_dim=[1],
                hidden_layer_units=2,
                share_weights_in_stack=False,
                nb_harmonics=None,
            ),
        }
    elif name == "NHiTS":
        cases = {
            "normalized_max_pool": _nhits_case(
                module,
                batch_size=2,
                seq_len=8,
                pred_len=4,
                channels=2,
                stack_types=["identity", "identity"],
                n_blocks=[1, 1],
                mlp_units=[[7, 7]],
                n_pool_kernel_size=[2, 1],
                n_freq_downsample=[2, 1],
                pooling_mode="MaxPool1d",
                interpolation_mode="linear",
                dropout=0.0,
                activation="ReLU",
                use_norm=True,
            ),
            "raw_avg_pool_train_dropout": _nhits_case(
                module,
                batch_size=1,
                seq_len=5,
                pred_len=3,
                channels=3,
                stack_types=["identity"],
                n_blocks=[2],
                mlp_units=[[6, 6]],
                n_pool_kernel_size=[2],
                n_freq_downsample=[2],
                pooling_mode="AvgPool1d",
                interpolation_mode="nearest",
                dropout=0.2,
                activation="Tanh",
                use_norm=False,
            ),
            "minimum": _nhits_case(
                module,
                batch_size=1,
                seq_len=1,
                pred_len=1,
                channels=1,
                stack_types=["identity"],
                n_blocks=[1],
                mlp_units=[[2, 2]],
                n_pool_kernel_size=[1],
                n_freq_downsample=[1],
                pooling_mode="MaxPool1d",
                interpolation_mode="linear",
                dropout=0.0,
                activation="ReLU",
                use_norm=True,
            ),
        }
    else:
        raise KeyError(name)
    return {
        "schema_version": 1,
        "model": name,
        "passed": all(case["passed"] for case in cases.values()),
        "source": {**SOURCES[name], "files": [SOURCES[name]["file"]], "file_sha256": SOURCES[name]["sha256"]},
        "upstream_execution": "exact-pinned-source",
        "mapping_version": "basis-upstream-v1",
        "command": (
            "uv run python scripts/verify_basis_upstream_parity.py "
            "--nbeats-checkout <n-beats@06a4e209ada80bf1f403ced5228261784dfb26ed> "
            "--nhits-checkout <neuralforecast@6c4f3e557d0ed672314323edba972eb550cb3550>"
        ),
        "deterministic": {"seed": SEED, "device": "cpu"},
        "tolerances": {"atol": ATOL, "rtol": RTOL},
        "cases": cases,
    }


def _errors(detail: dict[str, Any], group: str) -> tuple[float, float]:
    values = [
        result
        for case in detail["cases"].values()
        for mode in case["report"]["modes"].values()
        for result in mode[group].values()
    ]
    return (
        max(float(item["max_abs"]) for item in values),
        max(float(item["max_rel"]) for item in values),
    )


def _check(passed: bool, evidence: list[str], **metrics):
    return {"passed": passed, "evidence": evidence, "metrics": metrics}


def canonical_result(name: str, detail: dict[str, Any], detail_path: Path) -> dict[str, Any]:
    fields = next(item for item in model_records(ROOT) if item["name"] == name)
    relative = detail_path.relative_to(ROOT).as_posix()
    output_abs, output_rel = _errors(detail, "outputs")
    intermediate_abs, intermediate_rel = _errors(detail, "intermediates")
    input_abs, input_rel = _errors(detail, "input_gradients")
    parameter_abs, parameter_rel = _errors(detail, "parameter_gradients")
    serial_passed = all(
        item["passed"]
        for case in detail["cases"].values()
        for item in case["serialization"].values()
    )
    passed = bool(detail["passed"] and serial_passed)
    evidence = [relative, "tests/test_basis_upstream_parity.py"]
    state_maps = [case["state_map"] for case in detail["cases"].values()]
    parameter_count = max(
        sum(1 for key in mapping if not key.endswith(("basis", "running_mean", "running_var")))
        for mapping in state_maps
    )
    buffer_count = max(
        sum(1 for key in mapping if key.endswith("basis")) for mapping in state_maps
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
            "uv run python -m unittest tests.test_basis_upstream_parity -v",
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
                "seed": SEED,
                "algorithms": torch.are_deterministic_algorithms_enabled(),
                "num_threads": torch.get_num_threads(),
            },
        },
        "passed": passed,
        "source": {
            key: detail["source"][key] for key in ("url", "revision", "license")
        },
        "mapping": {
            "version": detail["mapping_version"],
            "parameters": parameter_count,
            "buffers": buffer_count,
        },
        "fixture": {
            "identifier": "basis-model-modes-and-boundaries-v1",
            "description": (
                "Seeded float32 CPU cases cover eval/train, batch and channel boundaries, "
                "minimum sequence/horizon, basis or interpolation branches, and preprocessing."
            ),
        },
        "tolerances": detail["tolerances"],
        "modes": ["eval", "train"],
        "checks": {
            "outputs": _check(passed, evidence, max_abs=output_abs, max_rel=output_rel),
            "intermediates": _check(passed, evidence, max_abs=intermediate_abs, max_rel=intermediate_rel),
            "input_gradients": _check(passed, evidence, max_abs=input_abs, max_rel=input_rel),
            "parameter_gradients": _check(passed, evidence, max_abs=parameter_abs, max_rel=parameter_rel),
            "train_eval": _check(passed, evidence, modes="eval,train"),
            "buffers": _check(passed, evidence, mapped_buffers=buffer_count),
            "serialization": _check(serial_passed, evidence, max_abs=0.0),
            "preprocessing": _check(
                passed,
                evidence,
                contract=("channel-folding" if name == "NBeats" else "channel-folding+optional-instance-normalization"),
            ),
            "boundaries": _check(passed, evidence, cases=len(detail["cases"])),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nbeats-checkout", type=Path, required=True)
    parser.add_argument("--nhits-checkout", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "verification" / "parity")
    parser.add_argument("--index", type=Path, default=ROOT / DEFAULT_INDEX)
    args = parser.parse_args()
    modules = {
        "NBeats": load_nbeats(args.nbeats_checkout),
        "NHiTS": load_nhits(args.nhits_checkout),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failed = False
    for name, module in modules.items():
        torch.manual_seed(SEED)
        detail = verify_model(name, module)
        detail_path = args.output_dir / f"{name}.json"
        detail_path.write_text(json.dumps(detail, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        write_verification_result(args.index, canonical_result(name, detail, detail_path))
        print(f"{name}: {'PASS' if detail['passed'] else 'FAIL'}")
        failed = failed or not detail["passed"]
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
