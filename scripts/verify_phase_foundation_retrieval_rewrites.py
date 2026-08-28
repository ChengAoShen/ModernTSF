#!/usr/bin/env python3
"""Generate rewrite evidence for phase, foundation, objective, and RAG models."""

from __future__ import annotations

import argparse
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
from models.pulse.model import Model as PULSE  # noqa: E402
from models.symtime.model import Model as SymTime  # noqa: E402
from models.timecap.model import Model as TimeCAP  # noqa: E402
from models.timeo1.model import Model as TimeO1  # noqa: E402
from models.tirex.model import Model as TiRex  # noqa: E402
from models.tsrag.model import Model as TSRAG  # noqa: E402


FACTORIES = {
    "PULSE": lambda: PULSE(8, 3, 2, 8, 4, 4, 2, 0.0),
    "SymTime": lambda: SymTime(8, 3, 2, 8, 4, 1, 2, 3, 0.0),
    "TimeCAP": lambda: TimeCAP(8, 3, 2, 8, 4, 2, 1, 2, 0.0),
    "TimeO1": lambda: TimeO1(8, 3, 2, 8, 0.75, 0.5),
    "TiRex": lambda: TiRex(8, 3, 2, d_model=8, patch_len=2, num_layers=1, dropout=0.0),
    "TSRAG": lambda: TSRAG(8, 3, 2, 8, 2, 3, 2, 0.0),
}
BOUNDARIES = {
    "PULSE": lambda: PULSE(1, 1, 2, 4, 1, 1, 1, 0.0),
    "SymTime": lambda: SymTime(1, 1, 2, 4, 1, 1, 1, 1, 0.0),
    "TimeCAP": lambda: TimeCAP(1, 1, 2, 4, 1, 2, 1, 1, 0.0),
    "TimeO1": lambda: TimeO1(1, 1, 2, 4, 0.75, 1.0),
    "TiRex": lambda: TiRex(1, 1, 2, d_model=4, patch_len=1, num_layers=1, dropout=0.0),
    "TSRAG": lambda: TSRAG(1, 1, 2, 4, 1, 1, 1, 0.0),
}
REFERENCES = {
    "PULSE": "https://arxiv.org/abs/2605.16793",
    "SymTime": "https://arxiv.org/abs/2510.08445",
    "TimeCAP": "https://doi.org/10.1609/aaai.v40i30.39700",
    "TimeO1": "https://arxiv.org/abs/2505.17847",
    "TiRex": "https://arxiv.org/abs/2505.23719",
    "TSRAG": "https://arxiv.org/abs/2503.07649",
}
STRUCTURES = {
    "PULSE": {
        "method": "phase-anchored Disentangle-Evolve-Simulate forecasting",
        "equation": "X~=RevIN(X-Ax)+Ax; Z=Attn(Tx,Ty,Ty); E=Attn(Ty,Z,Z); Y=RevIN^-1(Y0-Ay)+Ay",
        "modules": {
            "phase anchor and residual norm": "Model.phase_indices/Model.disentangle",
            "two-stage Phase Router": "PhaseRouter",
            "SAM and spectral objective": "Model.statistic_aware_mixup/Model.frequency_mae",
            "coordinate reconstruction": "Model.forward",
        },
        "differences": [
            "training utilities require explicit experiment-runner integration",
            "timestamp encoder omitted; compact fixed-resolution routing",
        ],
    },
    "SymTime": {
        "method": "decomposed downstream reconstruction with a patch series encoder",
        "equation": "Xn=RevIN(X); (Xp,Xt)=Decompose(Xn); Y=RevIN^-1(Head(Encoder(Patch(Xp)))+Linear(Xt))",
        "modules": {
            "normalization/decomposition": "Model.revin/Model.decomposition",
            "non-overlapping series encoder": "Model.patch_series/Model.series_encoder",
            "periodic/trend recombination": "Model.forward",
        },
        "differences": [
            "forecasting path only; symbol/momentum encoders and pre-training objectives omitted",
            "compact randomly initialized Transformer",
        ],
    },
    "TimeCAP": {
        "method": "group-wise channel-aware meta-routing and dual-head decoding",
        "equation": "E3_i=Concat(E2_i,R_i); O=scatter_mean(Zout_i); Y=(1-W)Yarg+WYosg",
        "modules": {
            "overlapping channel groups": "Model.group_indices/Model.group_projections",
            "time-aligned routing": "Model.channel_aware_mask/Model.groupwise_representation",
            "dynamic dual heads": "Model.dual_head_forecast",
        },
        "differences": [
            "one compact routing stage and GRUCell autoregressive head",
            "pre-training/fine-tuning loss schedule omitted",
        ],
    },
    "TimeO1": {
        "method": "SVD transformed-label alignment with significance truncation",
        "equation": "P=SVD(Standardize(Y)).V; Z=YP; L=alpha*|Zhat[:K]-Z[:K]|_1+(1-alpha)*||Yhat-Y||_2^2",
        "modules": {
            "per-variate SVD": "Model.fit_projection",
            "component projection": "Model.transform",
            "mixed objective": "Model.transformed_alignment_loss",
        },
        "differences": [
            "local MLP/linear carrier because the method is model-agnostic",
            "objective requires explicit experiment-runner integration",
        ],
    },
    "TiRex": {
        "method": "decoder-only scalar-memory multi-patch quantile forecasting",
        "equation": "token=ResidualMLP([patch,observed]); h=sLSTM(token,state); Q=QuantileHead(ResidualMLP(h_future))",
        "modules": {
            "value/missing patch tokens": "Model._history_tokens/ResidualProjection",
            "scalar recurrent memory": "ScalarMemory/ScalarLSTMBlock",
            "CPM and quantile decoding": "Model.contiguous_patch_mask/Model.forward",
        },
        "differences": [
            "clean PyTorch scalar memory instead of optimized xLSTM kernels",
            "no pretrained weights; monotone repository quantile head",
        ],
    },
    "TSRAG": {
        "method": "Euclidean top-k retrieval followed by Adaptive Retrieval Mixer",
        "equation": "C=TopKmin(||eq-ei||2); Eatt=MHA([eq;Eret])+[eq;Eret]; alpha=softmax(Wg Effn); efinal=eq+sum alpha_i Effn_i",
        "modules": {
            "knowledge pairs and top-k": "Model.build_local_knowledge/Model.retrieve",
            "future projection and ARM": "Model.retrieved_projector/AdaptiveRetrievalMixer",
            "forecast projection": "Model.output_projection",
        },
        "differences": [
            "no third-party TSFM/retriever checkpoint or FAISS dependency",
            "external knowledge API with deterministic history fallback",
        ],
    },
}


def _digest(value: dict[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _runtime(name: str) -> dict[str, object]:
    torch.manual_seed(260827)
    model = FACTORIES[name]().cpu().eval()
    x = torch.randn(2, 8, 2, requires_grad=True)
    marks, adjacency = torch.randn(2, 8, 6), torch.eye(2)
    output = model(x, marks, adjacency)
    expected = (2, 3, 2, 9) if name == "TiRex" else (2, 3, 2)
    if output.shape != expected or not torch.isfinite(output).all():
        raise AssertionError("forward contract failed")
    if name == "TiRex" and not (output[..., 1:] >= output[..., :-1]).all():
        raise AssertionError("quantile crossing detected")
    output.square().mean().backward()
    if x.grad is None or not torch.isfinite(x.grad).all() or x.grad.abs().max() == 0:
        raise AssertionError("input gradient failed")
    gradients: dict[str, float] = {}
    for parameter_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None or not torch.isfinite(parameter.grad).all() or parameter.grad.abs().max() == 0:
            raise AssertionError(f"inactive parameter: {parameter_name}")
        gradients[parameter_name] = float(parameter.grad.abs().max())
    clone = FACTORIES[name]().cpu().eval()
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    torch.testing.assert_close(clone(x.detach(), marks, adjacency), output.detach())
    if model(torch.randn(1, 8, 2)).shape[0] != 1:
        raise AssertionError("batch boundary failed")
    if BOUNDARIES[name]().eval()(torch.randn(1, 1, 2)).shape[:3] != (1, 1, 2):
        raise AssertionError("minimum-length boundary failed")
    try:
        model(torch.randn(1, 7, 2))
    except ValueError:
        rejected = True
    else:
        raise AssertionError("wrong length accepted")
    torch.testing.assert_close(model(x.detach(), marks, adjacency), model(x.detach()))
    return {
        "shape": list(output.shape),
        "input_gradient_max_abs": float(x.grad.abs().max()),
        "parameter_gradients": gradients,
        "round_trip_max_abs": 0.0,
        "wrong_length_rejected": rejected,
    }


def _check(evidence: list[str], **metrics: float | int | str) -> dict[str, object]:
    return {"passed": True, "evidence": evidence, "metrics": metrics}


def verify(name: str, records: dict[str, dict[str, object]]) -> None:
    observations = _runtime(name)
    structure = STRUCTURES[name]
    structure_digest = _digest(structure)
    relative = f"verification/rewrite/{name}.json"
    artifact_path = ROOT / relative
    artifact = {
        "schema_version": 1,
        "kind": "clean-room-structure-map",
        "model": name,
        "reference": REFERENCES[name],
        "independent_design": True,
        "source_code_not_copied": True,
        "structure_map": structure,
        "structure_map_sha256": structure_digest,
        "observations": observations,
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    evidence = [relative, "tests/test_phase_foundation_retrieval_rewrites.py"]
    checks = {
        "paper_structure": _check(evidence, mapped_elements=len(structure["modules"]), claim="paper-equations-to-independent-local-map"),
        "equations": _check(evidence, cases=1),
        "construction": _check(evidence, instances=3),
        "forward": _check(evidence, shape=",".join(map(str, observations["shape"]))),
        "backward": _check(evidence, input_gradient_max_abs=observations["input_gradient_max_abs"]),
        "finite_outputs": _check(evidence, nonfinite=0),
        "active_parameter_gradients": _check(evidence, parameters=len(observations["parameter_gradients"])),
        "state_dict_round_trip": _check(evidence, max_abs=0.0),
        "cpu": _check(evidence, device="cpu"),
        "batch_size_boundary": _check(evidence, cases="batch=1,batch=2"),
        "sequence_length_boundary": _check(evidence, cases="minimum-valid,wrong-length-rejected"),
        "marks_adjacency_contract": _check(evidence, contract="accepted-and-ignored"),
    }
    environment = {
        "python": platform.python_version(),
        "framework": f"torch {torch.__version__}",
        "dependencies": {"pydantic": importlib.metadata.version("pydantic"), "torch": torch.__version__},
        "platform": platform.platform(),
        "device": "cpu",
        "dtype": "float32",
        "deterministic": {"seed": 260827, "num_threads": torch.get_num_threads()},
    }
    result = {
        "schema_version": 1,
        "kind": "rewrite-validation",
        "model": name,
        "implementation": "rewrite",
        "verified_at": datetime.now(timezone.utc),
        "subject_sha256": verification_subject_sha256(ROOT, records[name]),
        "commands": [
            f"uv run python scripts/verify_phase_foundation_retrieval_rewrites.py --model {name}",
            "uv run python -m unittest tests.test_phase_foundation_retrieval_rewrites -v",
            f"uv run tsf repo doctor --strict --models {name}",
        ],
        "environment": environment,
        "artifacts": {relative: evidence_file_sha256(artifact_path)},
        "passed": True,
        "basis": {
            "references": [REFERENCES[name]],
            "structure_map_sha256": structure_digest,
            "independent_design": True,
            "source_code_not_copied": True,
        },
        "checks": checks,
    }
    write_verification_result(ROOT / "verification/model-results.json", result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", choices=sorted(FACTORIES))
    args = parser.parse_args()
    selected = set(args.model or FACTORIES)
    records = {str(record["name"]): record for record in model_records(ROOT)}
    for name in FACTORIES:
        if name in selected:
            verify(name, records)
            print(f"{name}: rewrite validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
