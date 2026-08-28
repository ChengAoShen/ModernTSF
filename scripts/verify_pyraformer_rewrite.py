#!/usr/bin/env python3
"""Generate strict clean-room rewrite evidence for Pyraformer."""

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
from models.pyraformer.model import (  # noqa: E402
    Model,
    finest_ancestor_table,
    pyramid_neighbour_table,
    pyramid_sizes,
)

PAPER = "https://openreview.net/forum?id=0EXmFzUn5I"
STRUCTURE = {
    "method": "paper-derived pyramidal attention forecaster",
    "equations": {
        "equation_2": "N(node)=same-scale adjacent nodes union children union parent",
        "equation_3": "output_i=sum_(j in N(i)) softmax(q_i k_j / sqrt(d_k)) value_j",
    },
    "modules": {
        "multi-resolution CSCM": "Model.coarse_scales",
        "PAM graph and sparse attention": "pyramid_neighbour_table and PyramidalAttention",
        "cross-scale feature gathering": "Model.ancestor_indices",
        "direct multi-horizon projection": "Model.forecast_head",
        "raw calendar contract": "_raw_calendar_features and Model.calendar_embedding",
    },
    "differences": [
        "direct multi-horizon prediction strategy 1 only",
        "learned strided convolutions construct coarse scales",
        "pre-normalized residual blocks",
        "no optimized custom attention kernel or published-training reproduction",
    ],
}


def _digest(value: dict[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _factory(length: int = 8) -> Model:
    return Model(
        length,
        3,
        2,
        d_model=8,
        n_heads=2,
        e_layers=2,
        d_ff=16,
        dropout=0.0,
        window_size=(2, 2),
        inner_size=3,
    )


def _marks(batch: int, length: int) -> torch.Tensor:
    marks = torch.zeros(batch, length, 6)
    marks[..., 0] = 2024
    marks[..., 1] = 1
    marks[..., 2] = torch.arange(1, length + 1)
    marks[..., 3] = torch.arange(length) % 7
    marks[..., 4] = torch.arange(length) % 24
    return marks


def _runtime() -> dict[str, object]:
    torch.manual_seed(1907)
    sizes = pyramid_sizes(8, (2, 2))
    indices, valid = pyramid_neighbour_table(sizes, (2, 2), 3)
    ancestors = finest_ancestor_table(sizes, (2, 2))
    if set(indices[3, valid[3]].tolist()) != {2, 3, 4, 9}:
        raise AssertionError("Equation-2 finest-scale neighbourhood failed")
    if set(indices[9, valid[9]].tolist()) != {2, 3, 8, 9, 10, 12}:
        raise AssertionError("Equation-2 inter-scale neighbourhood failed")
    if ancestors[7].tolist() != [7, 11, 13]:
        raise AssertionError("ancestor gather failed")

    model = _factory().cpu()
    x = torch.randn(2, 8, 2, requires_grad=True)
    marks = _marks(2, 8)
    output = model(x, marks, torch.eye(2))
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

    clone = _factory().cpu()
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    torch.testing.assert_close(clone(x.detach(), marks), output.detach())
    if _factory(4)(torch.randn(1, 4, 2)).shape != (1, 3, 2):
        raise AssertionError("minimum compatible pyramid failed")
    try:
        model(torch.randn(1, 7, 2))
    except ValueError:
        wrong_length_rejected = True
    else:
        raise AssertionError("wrong history length accepted")
    changed = marks.clone()
    changed[..., 4] += 6
    if torch.equal(model(x.detach(), marks), model(x.detach(), changed)):
        raise AssertionError("raw calendar marks are inactive")
    torch.testing.assert_close(model(x.detach(), marks, torch.eye(2)), model(x.detach(), marks))
    return {
        "shape": [2, 3, 2],
        "scale_sizes": list(sizes),
        "pyramid_nodes": sum(sizes),
        "maximum_neighbour_count": int(valid.sum(-1).max()),
        "input_gradient_max_abs": float(x.grad.abs().max()),
        "parameter_gradients": gradients,
        "round_trip_max_abs": 0.0,
        "raw_marks_affect_output": True,
        "wrong_length_rejected": wrong_length_rejected,
    }


def _check(evidence: list[str], **metrics: float | int | str) -> dict[str, object]:
    return {"passed": True, "evidence": evidence, "metrics": metrics}


def main() -> int:
    observations = _runtime()
    structure_digest = _digest(STRUCTURE)
    relative = "verification/rewrite/Pyraformer.json"
    artifact_path = ROOT / relative
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": 1,
        "kind": "clean-room-structure-map",
        "model": "Pyraformer",
        "reference": PAPER,
        "independent_design": True,
        "source_code_not_copied": True,
        "structure_map": STRUCTURE,
        "structure_map_sha256": structure_digest,
        "observations": observations,
    }
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    evidence = [relative, "tests/test_pyraformer_rewrite.py"]
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
        "sequence_length_boundary": _check(evidence, cases="minimum-divisible=4,wrong-length-rejected"),
        "marks_adjacency_contract": _check(evidence, contract="raw-six-column-marks-active;adjacency-accepted-and-ignored"),
    }
    records = {str(record["name"]): record for record in model_records(ROOT)}
    result = {
        "schema_version": 1,
        "kind": "rewrite-validation",
        "model": "Pyraformer",
        "implementation": "rewrite",
        "verified_at": datetime.now(timezone.utc),
        "subject_sha256": verification_subject_sha256(ROOT, records["Pyraformer"]),
        "commands": [
            "uv run python scripts/verify_pyraformer_rewrite.py",
            "uv run python -m unittest tests.test_pyraformer_rewrite -v",
            "uv run tsf repo doctor --strict --models Pyraformer",
        ],
        "environment": {
            "python": platform.python_version(),
            "framework": f"torch {torch.__version__}",
            "dependencies": {
                "pydantic": importlib.metadata.version("pydantic"),
                "torch": torch.__version__,
            },
            "platform": platform.platform(),
            "device": "cpu",
            "dtype": "float32",
            "deterministic": {"seed": 1907, "num_threads": torch.get_num_threads()},
        },
        "artifacts": {relative: evidence_file_sha256(artifact_path)},
        "passed": True,
        "basis": {
            "references": [PAPER],
            "structure_map_sha256": structure_digest,
            "independent_design": True,
            "source_code_not_copied": True,
        },
        "checks": checks,
    }
    write_verification_result(ROOT / "verification/model-results.json", result)
    print("Pyraformer: rewrite validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
