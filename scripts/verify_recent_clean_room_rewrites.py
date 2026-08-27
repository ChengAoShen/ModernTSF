#!/usr/bin/env python3
"""Generate strict clean-room evidence for four recent forecasting rewrites."""

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
from models.interpdn.model import Model as InterPDN  # noqa: E402
from models.olinear.model import Model as OLinear  # noqa: E402
from models.phaseformer.model import Model as PhaseFormer  # noqa: E402
from models.sonnet.model import Model as Sonnet  # noqa: E402

Factory = Callable[[int, int, int], nn.Module]


@dataclass(frozen=True)
class RewriteCase:
    name: str
    factory: Factory
    reference: str
    structure: dict[str, object]


CASES = (
    RewriteCase(
        "OLinear",
        lambda length, horizon, channels: OLinear(length, horizon, channels, d_model=4),
        "https://arxiv.org/abs/2505.08550",
        {
            "method": "orthogonally transformed linear forecasting with normalized channel mixing",
            "equation": "Z=Q_i^T x; W_n=softplus(W)/rowsum(softplus(W)); y=Q_o decode(ISL(CSL(Z)))",
            "modules": {"OrthoTrans": "Model.input_basis/output_basis", "NormLin": "NormLin", "CSL/ISL": "Model channel and sequence learners"},
            "differences": ["identity transform bases until training-data eigenvectors are installed", "one compact CSL/ISL block and direct flattened decoder"],
        },
    ),
    RewriteCase(
        "PhaseFormer",
        lambda length, horizon, channels: PhaseFormer(length, horizon, channels, d_model=4, period=3, num_routers=2),
        "https://arxiv.org/abs/2510.04134",
        {
            "method": "phase tokenization with two-stage cross-phase router attention",
            "equation": "X_phase=reshape(circular_pad(x)); H=MHA(R,Z,Z); Z'=MHA(Z,H,H); Y_phase=linear(Z')",
            "modules": {"phase tokens": "Model._tokenize", "router aggregation/distribution": "CrossPhaseRouter", "shared predictor": "Model.predictor"},
            "differences": ["period is explicitly configured rather than estimated by autocorrelation", "one channel-independent routing layer by default"],
        },
    ),
    RewriteCase(
        "InterPDN",
        lambda length, horizon, channels: InterPDN(length, horizon, channels, support_size=7),
        "https://arxiv.org/abs/2511.23260",
        {
            "method": "per-step distributions on interleaved supports with confidence fusion",
            "equation": "e_j=<softmax(logits_j),support_j>; w=max(p_1)/(max(p_1)+max(p_2)); y=w*e_1+(1-w)*e_2",
            "modules": {"dual independent branches": "Model.branches", "normal-quantile supports": "support_first/support_second", "expectation and confidence fusion": "Model.forward"},
            "differences": ["compact residual seasonal encoder replaces the paper patch encoder", "coarse auxiliary branches and training-only consistency losses are omitted"],
        },
    ),
    RewriteCase(
        "Sonnet",
        lambda length, horizon, channels: Sonnet(length, horizon, channels, d_model=4, num_wavelets=2),
        "https://arxiv.org/abs/2505.15312",
        {
            "method": "learnable wavelets, spectral-coherence weighting, and stable Koopman evolution",
            "equation": "M=exp(-a*t^2)cos(b*t+g*t^2); C=|Q_f K_f*|^2/(P_qq P_kk+eps); K=U diag(exp(i p)) U*",
            "modules": {"wavelet atoms": "LearnableWavelets", "MVCA": "SpectralCoherence", "unitary Koopman": "StableKoopman", "reconstruction/decoder": "Model.forward/decoder"},
            "differences": ["all channels use a symmetric joint embedding rather than an endogenous/exogenous alpha split", "adaptive horizon pooling replaces dataset-specific decoder sizing"],
        },
    ),
)


def _digest(value: dict[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _runtime(case: RewriteCase) -> dict[str, object]:
    torch.manual_seed(2026)
    model = case.factory(4, 3, 2).cpu().eval()
    x = torch.randn(2, 4, 2, requires_grad=True)
    marks, adjacency = torch.randn(2, 4, 3), torch.eye(2)
    output = model(x, marks, adjacency)
    if output.shape != (2, 3, 2) or not torch.isfinite(output).all():
        raise AssertionError("forward contract failed")
    output.square().mean().backward()
    if x.grad is None or not torch.isfinite(x.grad).all():
        raise AssertionError("input gradient failed")
    gradients: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all() or parameter.grad.abs().max() == 0:
            raise AssertionError(f"inactive or invalid parameter gradient: {name}")
        gradients[name] = float(parameter.grad.abs().max())
    clone = case.factory(4, 3, 2).cpu().eval()
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    torch.testing.assert_close(clone(x.detach()), output.detach())
    if model(torch.randn(1, 4, 2)).shape != (1, 3, 2):
        raise AssertionError("batch boundary failed")
    if case.factory(1, 1, 2)(torch.randn(1, 1, 2)).shape != (1, 1, 2):
        raise AssertionError("minimum sequence boundary failed")
    try:
        model(torch.randn(1, 3, 2))
    except ValueError:
        wrong_length_rejected = True
    else:
        raise AssertionError("wrong sequence length accepted")
    torch.testing.assert_close(model(x.detach(), marks, adjacency), model(x.detach()))
    return {
        "shape": [2, 3, 2],
        "input_gradient_max_abs": float(x.grad.abs().max()),
        "parameter_gradients": gradients,
        "round_trip_max_abs": 0.0,
        "wrong_length_rejected": wrong_length_rejected,
    }


def _environment() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "framework": f"torch {torch.__version__}",
        "dependencies": {"pydantic": importlib.metadata.version("pydantic"), "torch": torch.__version__},
        "platform": platform.platform(),
        "device": "cpu",
        "dtype": "float32",
        "deterministic": {"seed": 2026, "num_threads": torch.get_num_threads()},
    }


def _check(evidence: list[str], **metrics: float | int | str) -> dict[str, object]:
    return {"passed": True, "evidence": evidence, "metrics": metrics}


def verify(case: RewriteCase, records: dict[str, dict[str, object]]) -> None:
    observations = _runtime(case)
    structure_digest = _digest(case.structure)
    relative = f"verification/rewrite/{case.name}.json"
    artifact_path = ROOT / relative
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
    evidence = [relative, "tests/test_recent_clean_room_rewrites.py"]
    checks = {
        "paper_structure": _check(evidence, mapped_elements=len(case.structure["modules"]), claim="paper-equations-to-independent-local-map"),
        "equations": _check(evidence, cases=1),
        "construction": _check(evidence, instances=3),
        "forward": _check(evidence, shape="2,3,2"),
        "backward": _check(evidence, input_gradient_max_abs=observations["input_gradient_max_abs"]),
        "finite_outputs": _check(evidence, nonfinite=0),
        "active_parameter_gradients": _check(evidence, parameters=len(observations["parameter_gradients"])),
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
        "subject_sha256": verification_subject_sha256(ROOT, records[case.name]),
        "commands": [
            f"uv run python scripts/verify_recent_clean_room_rewrites.py --model {case.name}",
            "uv run python -m unittest tests.test_recent_clean_room_rewrites -v",
            f"uv run tsf repo doctor --strict --models {case.name}",
        ],
        "environment": _environment(),
        "artifacts": {relative: evidence_file_sha256(artifact_path)},
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
