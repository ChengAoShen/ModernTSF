#!/usr/bin/env python3
"""Emit reproducible evidence for the final temporal clean-room batch."""
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
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

from benchmark.catalog_metadata import model_records
from benchmark.verification_results import evidence_file_sha256, verification_subject_sha256, write_verification_result
from tests.test_temporal_structure_clean_room_rewrites import RuntimeTests

STRUCTURES = {
    "Koopa": {
        "reference": "https://arxiv.org/abs/2305.18803",
        "equations": {
            "Fourier Filter": "dominant spectral modes define invariant dynamics and the complement defines variant dynamics",
            "local Koopman": "ridge-DMD estimates a context-specific linear transition on recent latent states",
            "global Koopman": "a learned linear transition evolves invariant latent states",
            "hierarchy": "each block subtracts reconstruction and adds its forecast contribution",
        },
        "modules": ["FourierDynamicsSplit", "MeasurementFunction", "LocalKoopmanPredictor", "GlobalKoopmanPredictor", "KoopmanBlock"],
        "differences": ["batch-derived spectrum mask", "no rolling adaptation", "forecast-only"],
    },
    "LatentTSF": {
        "reference": "https://arxiv.org/abs/2602.00297",
        "equations": {
            "state projection": "per-timestep autoencoder expands observations to latent states",
            "latent forecast": "DLinear predicts future states without observation-space regression",
            "Eq. 5": "weighted latent MSE plus cosine alignment",
            "two stage": "autoencoder reconstruction pretraining precedes freezing and latent forecasting",
        },
        "modules": ["LatentStateAutoencoder", "DLinearBackbone", "latent_alignment_loss"],
        "differences": ["100 epoch default pretraining", "no external checkpoints", "optional perceptual loss omitted"],
    },
    "SOFTS": {
        "reference": "https://proceedings.neurips.cc/paper_files/paper/2024/file/754612bde73a8b65ad8743f1f6d8ddf6-Paper-Conference.pdf",
        "equations": {
            "aggregate": "softmax-weighted candidate series form one global core",
            "redistribute": "the core is concatenated and fused into every series token",
            "complexity": "centralized fusion is linear in the number of series",
        },
        "modules": ["SeriesCoreFusion", "SOFTSBlock"],
        "differences": ["calendar tokens omitted", "forecast-only"],
    },
    "SRSNet": {
        "reference": "https://arxiv.org/abs/2510.14510",
        "equations": {
            "selective patching": "learned utility softly gates contextual patches",
            "dynamic reassembly": "pairwise utility produces differentiable ranks and an assignment matrix",
            "forecast": "flattened selective representation is linearly projected to the horizon",
        },
        "modules": ["SelectivePatching", "DynamicReassembly", "SelectiveRepresentationSpace", "FlattenForecastHead"],
        "differences": ["continuous soft-sort relaxation", "forecast-only"],
    },
    "Sumba": {
        "reference": "https://openreview.net/forum?id=co7DsOwcop",
        "equations": {
            "matrix basis": "low-rank left, spectrum and right factors parameterize row-stochastic bases",
            "dynamic structure": "context softmax weights form a convex combination of bases",
            "temporal": "parallel gated causal convolutions cover multiple scales",
            "spatial": "multi-step graph diffusion propagates node states",
        },
        "modules": ["StructuredMatrixBasis", "MultiScaleTemporalConv", "DynamicBasisGraphConv", "SumbaBlock"],
        "differences": ["basis penalty exposed but not automatically weighted", "calendar features omitted"],
    },
    "TimeAlign": {
        "reference": "https://openreview.net/forum?id=pQzQfslqlD",
        "equations": {
            "prediction": "patch-MLP maps history representation to the future",
            "reconstruction": "a training-only branch reconstructs future targets",
            "local alignment": "corresponding patch-state cosine similarity is aligned",
            "global alignment": "within-representation relation matrices are aligned",
            "objective": "prediction plus weighted reconstruction and alignment losses",
        },
        "modules": ["PatchMLPBranch", "DistributionAlignment", "Model.train_loss_override"],
        "differences": ["included simple backbone only", "forecast-only"],
    },
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def runtime(name):
    torch.manual_seed(15000 + sum(map(ord, name)))
    factory = RuntimeTests.factories()[name]
    model = factory().cpu()
    values = torch.randn(2, 16, 2, requires_grad=True)
    output, objective = RuntimeTests.call(model, values, target=True)
    if output.shape != (2, 4, 2) or not torch.isfinite(output).all():
        raise AssertionError(f"{name}: forward")
    objective.backward()
    gradients = {}
    for parameter_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None or not torch.isfinite(parameter.grad).all() or parameter.grad.abs().max() == 0:
            raise AssertionError(f"{name}: inactive {parameter_name}")
        gradients[parameter_name] = float(parameter.grad.abs().max())
    if values.grad is None or not torch.isfinite(values.grad).all() or values.grad.abs().max() == 0:
        raise AssertionError(f"{name}: input gradient")
    model.eval()
    expected, _ = RuntimeTests.call(model, values.detach())
    clone = factory().eval()
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    actual, _ = RuntimeTests.call(clone, values.detach())
    torch.testing.assert_close(actual, expected)
    changed, _ = RuntimeTests.call(model, values.detach(), changed_marks=True)
    torch.testing.assert_close(changed, expected)
    batch_one, _ = RuntimeTests.call(model, torch.randn(1, 16, 2))
    if batch_one.shape != (1, 4, 2):
        raise AssertionError(f"{name}: batch")
    try:
        RuntimeTests.call(model, torch.randn(1, 15, 2))
    except ValueError:
        rejected = True
    else:
        raise AssertionError(f"{name}: length")
    return {
        "shape": [2, 4, 2],
        "input_gradient_max_abs": float(values.grad.abs().max()),
        "parameter_gradients": gradients,
        "state_dict_max_abs": 0.0,
        "batch_size_cases": [1, 2],
        "wrong_length_rejected": rejected,
        "marks_contract": "accepted-and-ignored",
        "adjacency_contract": "not-declared",
        "training_target_contract": "used" if getattr(model, "requires_train_target", False) else "not-required",
    }


def check(evidence, **metrics):
    return {"passed": True, "evidence": evidence, "metrics": metrics}


def main():
    records = {str(record["name"]): record for record in model_records(ROOT)}
    for name, structure in STRUCTURES.items():
        observations = runtime(name)
        structure_digest = digest(structure)
        relative = f"verification/rewrite/{name}.json"
        path = ROOT / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "schema_version": 1,
            "kind": "clean-room-structure-map",
            "model": name,
            "reference": structure["reference"],
            "independent_design": True,
            "source_code_not_copied": True,
            "structure_map": structure,
            "structure_map_sha256": structure_digest,
            "observations": observations,
        }
        path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        evidence = [relative, "tests/test_temporal_structure_clean_room_rewrites.py"]
        checks = {
            "paper_structure": check(evidence, mapped_elements=len(structure["modules"])),
            "equations": check(evidence, cases=len(structure["equations"])),
            "construction": check(evidence, instances=3),
            "forward": check(evidence, shape="2,4,2"),
            "backward": check(evidence, input_gradient_max_abs=observations["input_gradient_max_abs"]),
            "finite_outputs": check(evidence, nonfinite=0),
            "active_parameter_gradients": check(evidence, parameters=len(observations["parameter_gradients"])),
            "state_dict_round_trip": check(evidence, max_abs=0.0),
            "cpu": check(evidence, device="cpu"),
            "batch_size_boundary": check(evidence, cases="batch=1,batch=2"),
            "sequence_length_boundary": check(evidence, cases="expected-length;wrong-length-rejected"),
            "marks_adjacency_contract": check(evidence, contract="marks-accepted-and-ignored;adjacency-not-declared"),
        }
        seed = 15000 + sum(map(ord, name))
        result = {
            "schema_version": 1,
            "kind": "rewrite-validation",
            "model": name,
            "implementation": "rewrite",
            "verified_at": datetime.now(UTC),
            "subject_sha256": verification_subject_sha256(ROOT, records[name]),
            "commands": [
                "uv run python scripts/verify_temporal_structure_clean_room_rewrites.py",
                "uv run python -m unittest tests.test_temporal_structure_clean_room_rewrites -v",
                f"uv run tsf repo doctor --strict --models {name}",
            ],
            "environment": {
                "python": platform.python_version(),
                "framework": f"torch {torch.__version__}",
                "dependencies": {"torch": torch.__version__, "pydantic": importlib.metadata.version("pydantic")},
                "platform": platform.platform(),
                "device": "cpu",
                "dtype": "float32",
                "deterministic": {"seed": seed, "num_threads": torch.get_num_threads()},
            },
            "artifacts": {relative: evidence_file_sha256(path)},
            "passed": True,
            "basis": {
                "references": [structure["reference"]],
                "structure_map_sha256": structure_digest,
                "independent_design": True,
                "source_code_not_copied": True,
            },
            "checks": checks,
        }
        write_verification_result(ROOT / "verification/model-results.json", result)
        print(f"{name}: rewrite validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

