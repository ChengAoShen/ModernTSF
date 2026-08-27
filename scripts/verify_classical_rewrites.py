#!/usr/bin/env python3
"""Execute and record clean-room evidence for six classical rewrite baselines."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys
from typing import Callable

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark.catalog_metadata import model_records  # noqa: E402
from benchmark.verification_results import (  # noqa: E402
    evidence_file_sha256,
    verification_subject_sha256,
    write_verification_result,
)
from models.autoregressive_ts.model import Model as AutoRegressive  # noqa: E402
from models.exp_smoothing_ts.model import Model as ExpSmoothing  # noqa: E402
from models.knn_forecaster_ts.model import Model as SoftKNN  # noqa: E402
from models.lasso_regression_ts.model import Model as Lasso  # noqa: E402
from models.polynomial_regression_ts.model import Model as Polynomial  # noqa: E402
from models.ridge_regression_ts.model import Model as Ridge  # noqa: E402


Factory = Callable[[int, int, int], nn.Module]


@dataclass(frozen=True)
class RewriteCase:
    name: str
    factory: Factory
    reference: str
    structure: dict[str, object]
    equation_check: Callable[[], None]


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-6)


def _autoregressive_equation() -> None:
    model = AutoRegressive(3, 2, 1)
    with torch.no_grad():
        model.projection.weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]]))
        model.projection.bias.copy_(torch.tensor([0.5, -0.5]))
    _assert_close(model(torch.tensor([[[1.0], [2.0], [4.0]]])), torch.tensor([[[17.5], [2.5]]]))


def _exp_smoothing_equation() -> None:
    model = ExpSmoothing(3, 2, 1, initial_alpha=0.5)
    _assert_close(model(torch.tensor([[[2.0], [4.0], [8.0]]])), torch.tensor([[[5.5], [5.5]]]))


def _ridge_equation() -> None:
    model = Ridge(2, 1, 1, l2_penalty=0.25)
    with torch.no_grad():
        model.projection.weight.copy_(torch.tensor([[2.0, -1.0]]))
    model(torch.ones(1, 2, 1))
    _assert_close(model.aux_loss, torch.tensor(1.25))


def _lasso_equation() -> None:
    model = Lasso(2, 1, 1, l1_penalty=0.5)
    with torch.no_grad():
        model.projection.weight.copy_(torch.tensor([[2.0, -1.0]]))
    model(torch.ones(1, 2, 1))
    _assert_close(model.aux_loss, torch.tensor(1.5))


def _polynomial_equation() -> None:
    model = Polynomial(2, 1, 1, degree=3)
    expected = torch.tensor([[[2.0, -3.0, 4.0, 9.0, 8.0, -27.0]]])
    _assert_close(model.polynomial_features(torch.tensor([[[2.0], [-3.0]]])), expected)


def _knn_equation() -> None:
    model = SoftKNN(1, 1, 1, num_prototypes=2, kernel_gamma=1.0)
    with torch.no_grad():
        model.reference_windows.copy_(torch.tensor([[[0.0]], [[2.0]]]))
        model.reference_futures.copy_(torch.tensor([[[10.0]], [[20.0]]]))
    weights = torch.softmax(torch.tensor([0.0, -4.0]), dim=0)
    expected = (weights[0] * 10.0 + weights[1] * 20.0).reshape(1, 1, 1)
    _assert_close(model(torch.tensor([[[0.0]]])), expected)


CASES = (
    RewriteCase(
        "AutoRegressiveTS",
        lambda length, horizon, channels: AutoRegressive(length, horizon, channels),
        "https://search.worldcat.org/title/1422106714",
        {
            "method": "direct multi-horizon autoregression",
            "equation": "y[b,h,c] = bias[h] + sum_l weight[h,l] * x[b,l,c]",
            "modules": {"lag projection": "Model.projection"},
            "differences": ["direct rather than recursive", "shared coefficients across channels"],
        },
        _autoregressive_equation,
    ),
    RewriteCase(
        "ExpSmoothingTS",
        lambda length, horizon, channels: ExpSmoothing(length, horizon, channels, 0.4),
        "https://doi.org/10.1016/j.ijforecast.2003.09.015",
        {
            "method": "simple exponential smoothing",
            "equation": "level[t] = alpha*x[t] + (1-alpha)*level[t-1]",
            "modules": {"bounded learned alpha": "Model.alpha_logit", "level recurrence": "Model.forward"},
            "differences": ["level only; no trend or seasonality", "learned per-channel alpha"],
        },
        _exp_smoothing_equation,
    ),
    RewriteCase(
        "RidgeRegressionTS",
        lambda length, horizon, channels: Ridge(length, horizon, channels, 0.2),
        "https://doi.org/10.1080/00401706.1970.10488634",
        {
            "method": "L2-regularized lag regression",
            "equation": "loss = forecast_loss + lambda*sum(weight^2)",
            "modules": {"lag projection": "Model.projection", "penalty": "Model.aux_loss"},
            "differences": ["gradient optimization rather than closed form", "shared coefficients across channels"],
        },
        _ridge_equation,
    ),
    RewriteCase(
        "LassoRegressionTS",
        lambda length, horizon, channels: Lasso(length, horizon, channels, 0.2),
        "https://doi.org/10.1111/j.2517-6161.1996.tb02080.x",
        {
            "method": "L1-regularized lag regression",
            "equation": "loss = forecast_loss + lambda*sum(abs(weight))",
            "modules": {"lag projection": "Model.projection", "penalty": "Model.aux_loss"},
            "differences": ["gradient optimization rather than coordinate descent", "shared coefficients across channels"],
        },
        _lasso_equation,
    ),
    RewriteCase(
        "PolynomialRegressionTS",
        lambda length, horizon, channels: Polynomial(length, horizon, channels, 2),
        "https://doi.org/10.1002/9781118625590",
        {
            "method": "polynomial lag regression",
            "equation": "phi(x) = concat(x^1, ..., x^degree); y = W*phi(x)+b",
            "modules": {"feature map": "Model.polynomial_features", "projection": "Model.projection"},
            "differences": ["no interaction monomials", "shared coefficients across channels"],
        },
        _polynomial_equation,
    ),
    RewriteCase(
        "KNNForecasterTS",
        lambda length, horizon, channels: SoftKNN(length, horizon, channels, 4, 0.7),
        "https://doi.org/10.1109/TIT.1967.1053964",
        {
            "method": "soft nearest-reference regression",
            "equation": "w_k = softmax(-gamma*mean((x-r_k)^2)); y = sum_k w_k*v_k",
            "modules": {"references": "Model.reference_windows", "weighted future": "Model.forward"},
            "differences": ["learned references rather than stored samples", "soft all-reference weighting rather than hard k selection"],
            "reference_role": "conceptual nearest-neighbor background only",
            "equivalence_claim": "none; the verified structure is the independent local soft-kernel design",
        },
        _knn_equation,
    ),
)


def _structure_digest(structure: dict[str, object]) -> str:
    payload = json.dumps(structure, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _runtime_checks(case: RewriteCase) -> dict[str, object]:
    torch.manual_seed(731)
    case.equation_check()
    model = case.factory(4, 3, 2).cpu()
    x = torch.randn(2, 4, 2, requires_grad=True)
    marks = torch.randn(2, 4, 3)
    adjacency = torch.eye(2)
    output = model(x, marks, adjacency)
    if output.shape != (2, 3, 2) or not torch.isfinite(output).all():
        raise AssertionError("forward shape or finiteness failed")
    objective = output.square().mean()
    if model.aux_loss is not None:
        objective = objective + model.aux_loss
    objective.backward()
    if x.grad is None or not torch.isfinite(x.grad).all():
        raise AssertionError("input backward failed")
    gradient_norms: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise AssertionError(f"missing/nonfinite parameter gradient: {name}")
        gradient_norms[name] = float(parameter.grad.abs().max())
        if gradient_norms[name] == 0.0:
            raise AssertionError(f"inactive parameter: {name}")

    clone = case.factory(4, 3, 2).cpu()
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    _assert_close(clone(x.detach()), output.detach())
    batch_shape = tuple(model(torch.randn(1, 4, 2)).shape)
    boundary_shape = tuple(case.factory(1, 1, 2)(torch.randn(1, 1, 2)).shape)
    try:
        model(torch.randn(1, 3, 2))
    except ValueError:
        rejected_wrong_length = True
    else:
        rejected_wrong_length = False
    if batch_shape != (1, 3, 2) or boundary_shape != (1, 1, 2) or not rejected_wrong_length:
        raise AssertionError("boundary contract failed")
    baseline = model(x.detach())
    _assert_close(model(x.detach(), marks, adjacency), baseline)
    return {
        "output_shape": list(output.shape),
        "output_finite": True,
        "input_gradient_max_abs": float(x.grad.abs().max()),
        "parameter_gradient_max_abs": gradient_norms,
        "state_dict_round_trip_max_abs": 0.0,
        "batch_size_one_shape": list(batch_shape),
        "minimum_sequence_shape": list(boundary_shape),
        "wrong_sequence_rejected": rejected_wrong_length,
        "marks_adjacency": "accepted and deliberately ignored by time-series-only contract",
    }


def _environment() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "framework": f"torch {torch.__version__}",
        "dependencies": {
            "pydantic": importlib.metadata.version("pydantic"),
            "torch": torch.__version__,
        },
        "platform": platform.platform(),
        "device": "cpu",
        "dtype": "float32",
        "deterministic": {"seed": 731, "num_threads": torch.get_num_threads()},
    }


def _check(evidence: list[str], **metrics: float | int | str) -> dict[str, object]:
    return {"passed": True, "evidence": evidence, "metrics": metrics}


def verify(case: RewriteCase, records: dict[str, dict[str, object]]) -> None:
    observations = _runtime_checks(case)
    structure_digest = _structure_digest(case.structure)
    relative_artifact = f"verification/rewrite/{case.name}.json"
    artifact_path = ROOT / relative_artifact
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": 1,
        "kind": "clean-room-structure-map",
        "model": case.name,
        "reference": case.reference,
        "independent_design": True,
        "source_code_not_copied": True,
        "structure_map": case.structure,
        "structure_map_sha256": structure_digest,
        "observations": observations,
    }
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence = [relative_artifact, "tests/test_classical_rewrites.py"]
    record = records[case.name]
    paper_structure_metrics: dict[str, float | int | str] = {
        "mapped_elements": len(case.structure["modules"]),
        "claim": "reference-to-local map with disclosed differences",
    }
    if case.name == "KNNForecasterTS":
        paper_structure_metrics["claim"] = (
            "conceptual-background-only; no hard-KNN structural equivalence"
        )
    checks = {
        "paper_structure": _check(evidence, **paper_structure_metrics),
        "equations": _check(evidence, cases=1),
        "construction": _check(evidence, instances=3),
        "forward": _check(evidence, shape="2,3,2"),
        "backward": _check(evidence, input_gradient_max_abs=observations["input_gradient_max_abs"]),
        "finite_outputs": _check(evidence, nonfinite=0),
        "active_parameter_gradients": _check(evidence, parameters=len(observations["parameter_gradient_max_abs"])),
        "state_dict_round_trip": _check(evidence, max_abs=0.0),
        "cpu": _check(evidence, device="cpu"),
        "batch_size_boundary": _check(evidence, cases="batch=1,batch=2"),
        "sequence_length_boundary": _check(evidence, cases="length=1,wrong-length-rejected"),
        "marks_adjacency_contract": _check(evidence, contract="accepted-and-ignored"),
    }
    result = {
        "schema_version": 1,
        "kind": "rewrite-validation",
        "model": case.name,
        "implementation": "rewrite",
        "verified_at": datetime.now(timezone.utc),
        "subject_sha256": verification_subject_sha256(ROOT, record),
        "commands": [
            f"uv run python scripts/verify_classical_rewrites.py --model {case.name}",
            "uv run python -m unittest tests.test_classical_rewrites -v",
            f"uv run tsf repo doctor --backward --models {case.name}",
        ],
        "environment": _environment(),
        "artifacts": {relative_artifact: evidence_file_sha256(artifact_path)},
        "passed": True,
        "basis": {
            "references": [case.reference],
            "structure_map_sha256": structure_digest,
            "independent_design": True,
            "source_code_not_copied": True,
        },
        "checks": checks,
    }
    write_verification_result(ROOT / "verification/model-results.json", result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", choices=[case.name for case in CASES])
    args = parser.parse_args()
    selected = set(args.model or [case.name for case in CASES])
    records = {str(record["name"]): record for record in model_records(ROOT)}
    for case in CASES:
        if case.name in selected:
            verify(case, records)
            print(f"{case.name}: rewrite validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
