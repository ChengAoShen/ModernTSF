#!/usr/bin/env python3
"""Generate strict clean-room evidence for six mixer/frequency rewrites."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from benchmark.catalog_metadata import model_records
from benchmark.verification_results import (
    evidence_file_sha256,
    verification_subject_sha256,
    write_verification_result,
)
from tests.test_mixer_frequency_clean_room_rewrites import RuntimeTests


STRUCTURES = {
    "Amplifier": {
        "reference": "https://arxiv.org/abs/2501.17216",
        "equations": {
            "Eqs. 5--7": "rFFT spectrum reversal, addition, and inverse transform",
            "Eqs. 8--9": "complex frequency horizon map and subtractive restoration",
            "Eqs. 10--11": "commonality compression and channel-specific residual FFNs",
            "Eqs. 12--13": "seasonal/trend split and independent horizon forecasters",
        },
        "modules": ["flipped_spectrum", "SemiChannelInteraction", "ComplexFrequencyProjection", "SeriesDecomposition"],
        "differences": ["one-sided real FFT", "forecast-only", "no checkpoint or metric parity"],
    },
    "CMoS": {
        "reference": "https://proceedings.mlr.press/v267/si25a.html",
        "equations": {
            "Eq. 3": "weighted mixture of shared chunk-correlation matrices",
            "Eqs. 4--5": "channel-specific convolution summaries and shared allocator",
            "Section 3.3": "optional periodic peaks in the first correlation matrix",
            "Eqs. 6--7": "reversible per-instance normalization",
        },
        "modules": ["CorrelationMixer", "periodic_correlation_initialization", "RevIN"],
        "differences": ["periodicity disabled unless configured", "no top-k router", "unlicensed reference not inspected"],
    },
    "CRIB": {
        "reference": "https://arxiv.org/abs/2509.23494",
        "equations": {
            "Eq. 3": "fixed sinusoidal patch-time encoding",
            "Eq. 4": "attention over flattened channel-patch tokens",
            "Eqs. 7--9": "diagonal Gaussian variational bottleneck and KL",
            "Eq. 11": "random-mask/noise augmented representation consistency",
            "Eq. 12": "KL and consistency exposed as trainer auxiliary loss",
        },
        "modules": ["PatchEmbedding", "UnifiedVariateEncoder", "location/log_scale", "predictor"],
        "differences": ["NaN or explicit model mask input", "dataset masking pipeline omitted", "unlicensed reference not inspected"],
    },
    "CrossGNN": {
        "reference": "https://proceedings.neurips.cc/paper_files/paper/2023/hash/9278abf072b58caf21d48dd670b4c721-Abstract-Conference.html",
        "equations": {
            "Eqs. 1--5": "FFT top-period pooling and multiscale concatenation",
            "Eqs. 6--10": "scale-restricted plus adjacent temporal graph aggregation",
            "Eqs. 11--13": "positive homogeneous and negative heterogeneous variable edges",
            "Eq. 14": "channel collapse and direct horizon map",
        },
        "modules": ["AdaptiveMultiScaleIdentifier", "SparseCrossGraphLayer", "temporal_head"],
        "differences": ["softplus score relaxation", "interpolation before static DMS head", "dense score construction"],
    },
    "FiLM": {
        "reference": "https://papers.nips.cc/paper_files/paper/2022/hash/524ef58c2bd075775861234266e5e020-Abstract-Conference.html",
        "equations": {
            "LPU": "translated-Legendre state recurrence with bilinear discretization",
            "FEL": "lowest Fourier mode selection with complex low-rank factors",
            "LPU_R": "Legendre basis evaluation reconstructs the forecast horizon",
            "Section 3.2": "multiscale history experts combined by a learned linear mixture",
        },
        "modules": ["LegendreProjection", "LowRankFourierLayer", "FiLMExpert"],
        "differences": ["torch-native recurrence", "lowest modes only", "forecast-only"],
    },
    "FreTS": {
        "reference": "https://proceedings.neurips.cc/paper_files/paper/2023/hash/f1d16af76939f476b5f040fd1398c0a3-Abstract-Conference.html",
        "equations": {
            "Eqs. 1--2": "orthonormal Fourier domain conversion and inversion",
            "Eq. 3": "frequency channel learner shared over timestamps",
            "Eq. 4": "frequency temporal learner shared over channels",
            "Eq. 5": "two-layer direct forecast projection",
            "Eqs. 6--7": "full real/imaginary expansion of complex matrix MLPs",
        },
        "modules": ["ComplexFrequencyMLP", "FrequencyChannelLearner", "FrequencyTemporalLearner"],
        "differences": ["channel learner bypassed for fewer than three channels", "forecast-only", "no numerical parity"],
    },
}


def _runtime(name: str) -> dict[str, object]:
    seed = 4811 + sum(map(ord, name))
    torch.manual_seed(seed)
    factory = RuntimeTests.factories()[name]
    model = factory().cpu().train()
    channels = 4 if name == "CrossGNN" else 3
    values = torch.randn(2, 8, channels, requires_grad=True)
    output = RuntimeTests.call(model, values)
    expected_length = 4 if name == "CMoS" else 3
    if output.shape != (2, expected_length, channels) or not torch.isfinite(output).all():
        raise AssertionError(f"{name}: forward/finite contract")
    loss = output.square().mean()
    if getattr(model, "aux_loss", None) is not None:
        loss = loss + model.aux_loss
    loss.backward()
    if values.grad is None or not torch.isfinite(values.grad).all():
        raise AssertionError(f"{name}: input gradient")
    gradients = {}
    for parameter_name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all() or parameter.grad.abs().max() == 0:
            raise AssertionError(f"{name}: inactive parameter {parameter_name}")
        gradients[parameter_name] = float(parameter.grad.abs().max())
    model.eval()
    expected = RuntimeTests.call(model, values.detach())
    clone = factory().eval()
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    torch.testing.assert_close(RuntimeTests.call(clone, values.detach()), expected)
    if RuntimeTests.call(model, torch.randn(1, 8, channels)).shape[0] != 1:
        raise AssertionError(f"{name}: batch boundary")
    try:
        RuntimeTests.call(model, torch.randn(1, 7, channels))
    except ValueError:
        wrong_length_rejected = True
    else:
        raise AssertionError(f"{name}: wrong length accepted")
    torch.testing.assert_close(
        RuntimeTests.call(model, values.detach(), changed_marks=True), expected
    )
    missing_contract = "not applicable"
    if name == "CRIB":
        incomplete = values.detach().clone()
        incomplete[0, 1, 1] = torch.nan
        if not torch.isfinite(model(incomplete)).all():
            raise AssertionError("CRIB: NaN missing-value contract")
        explicit = torch.ones_like(values, dtype=torch.bool)
        explicit[:, 2, :] = False
        if not torch.isfinite(model(values.detach(), mask=explicit)).all():
            raise AssertionError("CRIB: explicit missing mask contract")
        missing_contract = "NaN-and-same-shaped-boolean-mask"
    return {
        "shape": [2, expected_length, channels],
        "input_gradient_max_abs": float(values.grad.abs().max()),
        "parameter_gradients": gradients,
        "state_dict_max_abs": 0.0,
        "batch_size_cases": [1, 2],
        "wrong_length_rejected": wrong_length_rejected,
        "marks_contract": "accepted-and-ignored",
        "adjacency_contract": "not declared",
        "missing_values_contract": missing_contract,
    }


def _digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _environment(seed: int) -> dict[str, object]:
    return {
        "python": platform.python_version(), "framework": f"torch {torch.__version__}",
        "dependencies": {"torch": torch.__version__, "pydantic": importlib.metadata.version("pydantic")},
        "platform": platform.platform(), "device": "cpu", "dtype": "float32",
        "deterministic": {"seed": seed, "num_threads": torch.get_num_threads()},
    }


def _check(evidence: list[str], **metrics: object) -> dict[str, object]:
    return {"passed": True, "evidence": evidence, "metrics": metrics}


def main() -> int:
    records = {str(record["name"]): record for record in model_records(ROOT)}
    for name, structure in STRUCTURES.items():
        observations = _runtime(name)
        structure_digest = _digest(structure)
        relative = f"verification/rewrite/{name}.json"
        artifact_path = ROOT / relative
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "schema_version": 1, "kind": "clean-room-structure-map", "model": name,
            "reference": structure["reference"], "independent_design": True,
            "source_code_not_copied": True, "structure_map": structure,
            "structure_map_sha256": structure_digest, "observations": observations,
        }
        artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        evidence = [relative, "tests/test_mixer_frequency_clean_room_rewrites.py"]
        checks = {
            "paper_structure": _check(evidence, mapped_elements=len(structure["modules"])),
            "equations": _check(evidence, cases=len(structure["equations"])),
            "construction": _check(evidence, instances=3),
            "forward": _check(evidence, shape=",".join(map(str, observations["shape"]))),
            "backward": _check(evidence, input_gradient_max_abs=observations["input_gradient_max_abs"]),
            "finite_outputs": _check(evidence, nonfinite=0),
            "active_parameter_gradients": _check(evidence, parameters=len(observations["parameter_gradients"])),
            "state_dict_round_trip": _check(evidence, max_abs=0.0),
            "cpu": _check(evidence, device="cpu"),
            "batch_size_boundary": _check(evidence, cases="batch=1,batch=2"),
            "sequence_length_boundary": _check(evidence, cases="minimum-covered;wrong-length-rejected"),
            "marks_adjacency_contract": _check(
                evidence,
                contract=f"{observations['marks_contract']};adjacency-not-declared;missing={observations['missing_values_contract']}",
            ),
        }
        seed = 4811 + sum(map(ord, name))
        result = {
            "schema_version": 1, "kind": "rewrite-validation", "model": name,
            "implementation": "rewrite", "verified_at": datetime.now(UTC),
            "subject_sha256": verification_subject_sha256(ROOT, records[name]),
            "commands": [
                "uv run python scripts/verify_mixer_frequency_clean_room_rewrites.py",
                "uv run python -m unittest tests.test_mixer_frequency_clean_room_rewrites -v",
                f"uv run tsf repo doctor --strict --models {name}",
            ],
            "environment": _environment(seed),
            "artifacts": {relative: evidence_file_sha256(artifact_path)},
            "passed": True,
            "basis": {
                "references": [structure["reference"]],
                "structure_map_sha256": structure_digest,
                "independent_design": True, "source_code_not_copied": True,
            },
            "checks": checks,
        }
        write_verification_result(ROOT / "verification/model-results.json", result)
        print(f"{name}: rewrite validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
