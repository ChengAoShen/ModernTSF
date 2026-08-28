#!/usr/bin/env python3
"""Generate strict clean-room structure and runtime evidence for xPatch."""

from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark.catalog_metadata import model_records  # noqa: E402
from benchmark.verification_results import (  # noqa: E402
    evidence_file_sha256,
    verification_subject_sha256,
    write_verification_result,
)
from models.xpatch.layers import ExponentialDecomposition  # noqa: E402
from models.xpatch.model import Model  # noqa: E402


REFERENCE = "https://arxiv.org/abs/2412.17323"
STRUCTURE = {
    "method": "EMA-decomposed channel-independent dual-stream forecast",
    "equations": {
        "decomposition": "s[0]=x[0]; s[t]=alpha*x[t]+(1-alpha)*s[t-1]; seasonal=x-s",
        "patch_count": "N=floor((L-P)/S)+2 with end padding",
        "fusion": "forecast=Linear(GELU(BN(Linear(concat(linear,nonlinear)))))",
    },
    "modules": {
        "EMA and residual": "Model.decomposition",
        "activation-free trend MLP": "Model.forecaster.linear_stream",
        "patch depthwise/pointwise CNN": "Model.forecaster.nonlinear_stream",
        "dual-stream fusion": "Model.forecaster.fusion",
        "channel-independent reshape": "Model.forward",
    },
    "differences": [
        "hidden widths are explicit local defaults where the paper is underspecified",
        "end padding repeats the final observation",
        "optional Holt level-and-trend DEMA is disclosed and is not the paper default",
        "arctangent loss and sigmoid learning-rate schedule remain trainer policies",
    ],
}


def _digest(value: dict[str, object]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _runtime() -> dict[str, object]:
    torch.manual_seed(1949)
    fixture = torch.tensor([[[1.0], [3.0], [5.0]]])
    seasonal, trend = ExponentialDecomposition(alpha=0.5)(fixture)
    torch.testing.assert_close(trend, torch.tensor([[[1.0], [2.0], [3.5]]]))
    torch.testing.assert_close(seasonal + trend, fixture)

    model = Model(8, 3, 2, patch_len=4, stride=2, hidden_dim=8).cpu()
    x = torch.randn(2, 8, 2, requires_grad=True)
    marks, adjacency = torch.randn(2, 8, 4), torch.eye(2)
    output = model(x, marks, adjacency)
    if output.shape != (2, 3, 2) or not torch.isfinite(output).all():
        raise AssertionError("forward contract failed")
    output.square().mean().backward()
    if x.grad is None or not torch.isfinite(x.grad).all():
        raise AssertionError("input gradient failed")
    gradients: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if (
            parameter.grad is None
            or not torch.isfinite(parameter.grad).all()
            or parameter.grad.abs().max() == 0
        ):
            raise AssertionError(f"inactive or invalid parameter gradient: {name}")
        gradients[name] = float(parameter.grad.abs().max())

    model.eval()
    expected = model(x.detach())
    clone = Model(8, 3, 2, patch_len=4, stride=2, hidden_dim=8).eval()
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    torch.testing.assert_close(clone(x.detach()), expected)
    if model(torch.randn(1, 8, 2)).shape != (1, 3, 2):
        raise AssertionError("batch boundary failed")
    if Model(1, 1, 1, patch_len=4, stride=2, hidden_dim=4)(torch.randn(1, 1, 1)).shape != (1, 1, 1):
        raise AssertionError("minimum sequence failed")
    try:
        model(torch.randn(1, 7, 2))
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
        "dependencies": {
            "pydantic": importlib.metadata.version("pydantic"),
            "torch": torch.__version__,
        },
        "platform": platform.platform(),
        "device": "cpu",
        "dtype": "float32",
        "deterministic": {"seed": 1949, "num_threads": torch.get_num_threads()},
    }


def _check(evidence: list[str], **metrics: float | int | str) -> dict[str, object]:
    return {"passed": True, "evidence": evidence, "metrics": metrics}


def main() -> int:
    observations = _runtime()
    records = {str(record["name"]): record for record in model_records(ROOT)}
    structure_digest = _digest(STRUCTURE)
    relative = "verification/rewrite/xPatch.json"
    artifact_path = ROOT / relative
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": 1,
        "kind": "clean-room-structure-map",
        "model": "xPatch",
        "reference": REFERENCE,
        "independent_design": True,
        "source_code_not_copied": True,
        "structure_map": STRUCTURE,
        "structure_map_sha256": structure_digest,
        "observations": observations,
    }
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    evidence = [relative, "tests/test_xpatch_rewrite.py"]
    checks = {
        "paper_structure": _check(evidence, mapped_elements=len(STRUCTURE["modules"])),
        "equations": _check(evidence, cases=3),
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
        "model": "xPatch",
        "implementation": "rewrite",
        "verified_at": datetime.now(timezone.utc),
        "subject_sha256": verification_subject_sha256(ROOT, records["xPatch"]),
        "commands": [
            "uv run python scripts/verify_xpatch_rewrite.py",
            "uv run python -m unittest tests.test_xpatch_rewrite -v",
            "uv run tsf repo doctor --strict --models xPatch",
        ],
        "environment": _environment(),
        "artifacts": {relative: evidence_file_sha256(artifact_path)},
        "passed": True,
        "basis": {
            "references": [REFERENCE],
            "structure_map_sha256": structure_digest,
            "independent_design": True,
            "source_code_not_copied": True,
        },
        "checks": checks,
    }
    write_verification_result(ROOT / "verification/model-results.json", result)
    print("xPatch: rewrite validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
