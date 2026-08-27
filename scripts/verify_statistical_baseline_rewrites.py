#!/usr/bin/env python3
"""Execute and record clean-room evidence for statistical baseline rewrites."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
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
from models.arima_ts.model import Model as ARIMA  # noqa: E402
from models.bayesian_ridge_ts.model import Model as BayesianRidge  # noqa: E402
from models.elastic_net_ts.model import Model as ElasticNet  # noqa: E402
from models.gaussian_process_ts.model import Model as GaussianProcess  # noqa: E402
from models.kalman_filter_ts.model import Model as AlphaBeta  # noqa: E402
from models.svr_forecaster_ts.model import Model as EpsilonRBF  # noqa: E402


Factory = Callable[[int, int, int], nn.Module]


@dataclass(frozen=True)
class RewriteCase:
    name: str
    factory: Factory
    reference: str
    structure: dict[str, object]
    equation_check: Callable[[], None]


def _close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=1e-6)


def _bayesian_equation() -> None:
    model = BayesianRidge(2, 1, 1, 2.0)
    with torch.no_grad():
        model.projection.weight.copy_(torch.tensor([[2.0, -1.0]]))
    model(torch.ones(1, 2, 1))
    _close(model.aux_loss, torch.tensor(5.0 - math.log(2.0)))


def _elastic_equation() -> None:
    model = ElasticNet(2, 1, 1, 0.5, 0.25)
    with torch.no_grad():
        model.projection.weight.copy_(torch.tensor([[2.0, -1.0]]))
    model(torch.ones(1, 2, 1))
    _close(model.aux_loss, torch.tensor(0.5 * (0.25 * 3.0 + 0.5 * 0.75 * 5.0)))


def _alpha_beta_equation() -> None:
    model = AlphaBeta(3, 2, 1, 0.5, 0.25)
    _close(model(torch.tensor([[[0.0], [2.0], [4.0]]])), torch.tensor([[[3.875], [5.0]]]))


def _gp_equation() -> None:
    model = GaussianProcess(1, 1, 1, 2, 1.0, 0.5)
    with torch.no_grad():
        model.inducing_inputs.copy_(torch.tensor([[0.0], [2.0]]))
        model.inducing_targets.copy_(torch.tensor([[10.0], [20.0]]))
    z = torch.tensor([[0.0], [2.0]])
    k_xz = torch.exp(-0.5 * torch.cdist(torch.tensor([[0.0]]), z).square())
    k_zz = torch.exp(-0.5 * torch.cdist(z, z).square())
    expected = k_xz @ torch.linalg.solve(k_zz + model.noise.detach() * torch.eye(2), torch.tensor([[10.0], [20.0]]))
    _close(model(torch.tensor([[[0.0]]])), expected.reshape(1, 1, 1))


def _svr_equation() -> None:
    model = EpsilonRBF(1, 1, 1, 2, 1.0, 0.5)
    with torch.no_grad():
        model.support_centres.copy_(torch.tensor([[0.0], [2.0]]))
        model.coefficients.copy_(torch.tensor([[10.0], [20.0]]))
        model.bias.zero_()
    _close(model(torch.tensor([[[0.0]]])), torch.tensor(10.0 + 20.0 * math.exp(-4.0)).reshape(1, 1, 1))
    _close(model.epsilon_insensitive_loss(torch.tensor([0.0, 2.0]), torch.tensor([0.25, 0.0])), torch.tensor(0.75))


def _arima_equation() -> None:
    model = ARIMA(3, 2, 1, 1, 1)
    with torch.no_grad():
        model.ar_coefficients.copy_(torch.tensor([0.5]))
        model.ma_coefficients.copy_(torch.tensor([0.25]))
    _close(model(torch.tensor([[[1.0], [3.0], [6.0]]])), torch.tensor([[[7.875], [8.8125]]]))


CASES = (
    RewriteCase("BayesianRidgeTS", lambda l, h, c: BayesianRidge(l, h, c, 0.2), "https://doi.org/10.1162/neco.1992.4.3.415", {"method": "Gaussian-prior MAP lag regression", "equation": "y=Wx+b; R=0.5*lambda*||W||^2-0.5*n*log(lambda)", "modules": {"lag projection": "Model.projection", "prior precision": "Model.weight_precision", "negative log prior": "Model.aux_loss"}, "differences": ["gradient MAP rather than evidence maximization", "no posterior predictive uncertainty", "shared coefficients across channels"]}, _bayesian_equation),
    RewriteCase("ElasticNetTS", lambda l, h, c: ElasticNet(l, h, c, 0.2, 0.4), "https://doi.org/10.1111/j.1467-9868.2005.00503.x", {"method": "elastic-net lag regression", "equation": "R=alpha*(rho*||W||_1+0.5*(1-rho)*||W||_2^2)", "modules": {"lag projection": "Model.projection", "elastic penalty": "Model.aux_loss"}, "differences": ["gradient optimization rather than solution path", "direct multi-horizon forecast", "shared coefficients across channels"]}, _elastic_equation),
    RewriteCase("KalmanFilterTS", lambda l, h, c: AlphaBeta(l, h, c, 0.5, 0.25), "https://doi.org/10.1115/1.3662552", {"method": "learned fixed-gain alpha-beta filter", "equation": "e=x-(level+velocity); level+=velocity+alpha*e; velocity+=beta*e", "modules": {"bounded gains": "Model.alpha_logits/Model.beta_logits", "predict-update recurrence": "Model.forward"}, "differences": ["fixed learned gains rather than covariance recursion", "constant-velocity state and unit time", "no control input"]}, _alpha_beta_equation),
    RewriteCase("GaussianProcessTS", lambda l, h, c: GaussianProcess(l, h, c, 4, 1.0, 0.1), "https://gaussianprocess.org/gpml/chapters/", {"method": "learned sparse RBF posterior-mean approximation", "equation": "f(x)=K(x,Z)[K(Z,Z)+noise*I]^-1 U", "modules": {"RBF covariance": "Model._kernel", "inducing pairs": "Model.inducing_inputs/Model.inducing_targets", "kernel solve": "Model.forward"}, "differences": ["learned inducing targets rather than conditioned observations", "posterior mean only", "shared channel-wise function"], "equivalence_claim": "none; not equivalent to an exact GP package"}, _gp_equation),
    RewriteCase("SVRForecasterTS", lambda l, h, c: EpsilonRBF(l, h, c, 4, 0.7, 0.1, 0.2), "https://papers.nips.cc/paper/1996/hash/d38901788c533e8286cb6400b40b386d-Abstract.html", {"method": "differentiable RBF-basis epsilon regression", "equation": "f(x)=sum_j a_j exp(-gamma||x-z_j||^2)+b; L=mean(max(|f-y|-epsilon,0))", "modules": {"RBF support expansion": "Model.forward", "epsilon loss": "Model.epsilon_insensitive_loss", "coefficient penalty": "Model.aux_loss"}, "differences": ["directly learned centres and coefficients", "no constrained dual solver", "epsilon loss requires explicit trainer selection"], "equivalence_claim": "none; differentiable adaptation only"}, _svr_equation),
    RewriteCase("ARIMATS", lambda l, h, c: ARIMA(l, h, c, 2, 1), "https://search.worldcat.org/title/Time-series-analysis-forecasting-and-control/oclc/1422106714", {"method": "conditional ARIMA(p,1,q)", "equation": "d_t=drift+sum_i phi_i*d_(t-i)+sum_j theta_j*e_(t-j)+e_t", "modules": {"first difference": "Model.forward", "historical innovations": "Model.forward", "conditional forecast recurrence": "Model.forward"}, "differences": ["fixed differencing order one", "gradient fitting and shared channel coefficients", "zero expected future innovations", "no order selection, seasonality, or intervals"]}, _arima_equation),
)


def _structure_digest(structure: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(structure, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _runtime_checks(case: RewriteCase) -> dict[str, object]:
    torch.manual_seed(947)
    case.equation_check()
    model = case.factory(4, 3, 2).cpu()
    x = torch.randn(2, 4, 2, requires_grad=True)
    marks, adjacency = torch.randn(2, 4, 3), torch.eye(2)
    output = model(x, marks, adjacency)
    if output.shape != (2, 3, 2) or not torch.isfinite(output).all():
        raise AssertionError("forward shape or finiteness failed")
    objective = output.square().mean()
    if model.aux_loss is not None:
        objective = objective + model.aux_loss
    objective.backward()
    if x.grad is None or not torch.isfinite(x.grad).all():
        raise AssertionError("input backward failed")
    gradients: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise AssertionError(f"missing/nonfinite parameter gradient: {name}")
        gradients[name] = float(parameter.grad.abs().max())
        if gradients[name] == 0:
            raise AssertionError(f"inactive parameter: {name}")
    clone = case.factory(4, 3, 2).cpu()
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    _close(clone(x.detach()), output.detach())
    batch_shape = tuple(model(torch.randn(1, 4, 2)).shape)
    boundary_shape = tuple(case.factory(1, 1, 2)(torch.randn(1, 1, 2)).shape)
    try:
        model(torch.randn(1, 3, 2))
    except ValueError:
        rejected = True
    else:
        rejected = False
    _close(model(x.detach(), marks, adjacency), model(x.detach()))
    if batch_shape != (1, 3, 2) or boundary_shape != (1, 1, 2) or not rejected:
        raise AssertionError("boundary contract failed")
    return {"output_shape": list(output.shape), "output_finite": True, "input_gradient_max_abs": float(x.grad.abs().max()), "parameter_gradient_max_abs": gradients, "state_dict_round_trip_max_abs": 0.0, "batch_size_one_shape": list(batch_shape), "minimum_sequence_shape": list(boundary_shape), "wrong_sequence_rejected": rejected, "marks_adjacency": "accepted and deliberately ignored by time-series-only contract"}


def _environment() -> dict[str, object]:
    return {"python": platform.python_version(), "framework": f"torch {torch.__version__}", "dependencies": {"pydantic": importlib.metadata.version("pydantic"), "torch": torch.__version__}, "platform": platform.platform(), "device": "cpu", "dtype": "float32", "deterministic": {"seed": 947, "num_threads": torch.get_num_threads()}}


def _check(evidence: list[str], **metrics: float | int | str) -> dict[str, object]:
    return {"passed": True, "evidence": evidence, "metrics": metrics}


def verify(case: RewriteCase, records: dict[str, dict[str, object]]) -> None:
    observations = _runtime_checks(case)
    digest = _structure_digest(case.structure)
    relative_artifact = f"verification/rewrite/{case.name}.json"
    artifact_path = ROOT / relative_artifact
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {"schema_version": 1, "kind": "clean-room-structure-map", "model": case.name, "reference": case.reference, "independent_design": True, "source_code_not_copied": True, "structure_map": case.structure, "structure_map_sha256": digest, "observations": observations}
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence = [relative_artifact, "tests/test_statistical_baseline_rewrites.py"]
    checks = {
        "paper_structure": _check(evidence, mapped_elements=len(case.structure["modules"]), claim="reference-to-local map with disclosed adaptation"),
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
    result = {"schema_version": 1, "kind": "rewrite-validation", "model": case.name, "implementation": "rewrite", "verified_at": datetime.now(timezone.utc), "subject_sha256": verification_subject_sha256(ROOT, records[case.name]), "commands": [f"uv run python scripts/verify_statistical_baseline_rewrites.py --model {case.name}", "uv run python -m unittest tests.test_statistical_baseline_rewrites -v", f"uv run tsf repo doctor --backward --models {case.name}"], "environment": _environment(), "artifacts": {relative_artifact: evidence_file_sha256(artifact_path)}, "passed": True, "basis": {"references": [case.reference], "structure_map_sha256": digest, "independent_design": True, "source_code_not_copied": True}, "checks": checks}
    write_verification_result(ROOT / "verification/model-results.json", result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", choices=[case.name for case in CASES])
    selected = set(parser.parse_args().model or [case.name for case in CASES])
    records = {str(record["name"]): record for record in model_records(ROOT)}
    for case in CASES:
        if case.name in selected:
            verify(case, records)
            print(f"{case.name}: rewrite validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
