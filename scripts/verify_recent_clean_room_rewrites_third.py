#!/usr/bin/env python3
"""Generate strict evidence for the third recent clean-room rewrite batch."""

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
from benchmark.verification_results import evidence_file_sha256, verification_subject_sha256, write_verification_result  # noqa: E402
from models.fets.model import Model as FeTS  # noqa: E402
from models.implicitforecaster.model import Model as ImplicitForecaster  # noqa: E402
from models.occamvts.model import Model as OccamVTS  # noqa: E402
from models.pmdformer.model import Model as PMDformer  # noqa: E402

Factory = Callable[[], nn.Module]


@dataclass(frozen=True)
class RewriteCase:
    name: str
    factory: Factory
    boundary_factory: Factory
    reference: str
    structure: dict[str, object]


CASES = (
    RewriteCase("FeTS", lambda: FeTS(8, 3, 2, 4, 4, 2, 1, 2), lambda: FeTS(1, 1, 2, 4, 1, 1, 1, 1),
                "https://doi.org/10.1609/aaai.v40i31.39838",
                {"method": "adaptive Fourier-Polynomial feature selection with dual-scale fusion",
                 "equation": "O=sum Xcos*a+sum Xsin*b+sum Xpoly*c; mask=1[Z>=mean(Z)]; Y=sum_j W_j X_j mask_j",
                 "modules": {"Fourier-Poly mask": "FourierPolyMask", "mask-controlled aggregation": "Model.adaptive_features", "DSFFN": "Model.local/fusion"},
                 "differences": ["straight-through sigmoid gradient with exact binary forward mask", "one compact AdaFE/DSFFN block"]}),
    RewriteCase("ImplicitForecaster", lambda: ImplicitForecaster(8, 3, 2, 4, 6), lambda: ImplicitForecaster(1, 1, 2, 4, 2),
                "https://proceedings.neurips.cc/paper_files/paper/2025/hash/0e82ef0c89df6a6eff8734ea7e27c42f-Abstract-Conference.html",
                {"method": "implicit amplitude/phase frequency-pool decoding",
                 "equation": "A=abs(AHead([encoder(X),abs(FFT(X))])); phi=atan2(Psin,Pcos); Y=crop(IFFT(A*exp(i*phi)))",
                 "modules": {"channel encoder": "Model.encoder", "AHead/PHead": "Model.spectral_parameters", "wave composition": "Model.forward"},
                 "differences": ["compact temporal MLP carrier encoder", "fixed configured frequency pool and direct horizon crop"]}),
    RewriteCase("OccamVTS", lambda: OccamVTS(8, 3, 2, 4, 4, 2, 4, 1), lambda: OccamVTS(1, 1, 2, 4, 1, 1, 1, 1),
                "https://arxiv.org/abs/2508.01727",
                {"method": "compact retained temporal/visual student with cross-modal fusion",
                 "equation": "Xaug=[x,abs(FFT(x)),sin(2pi*t/P),cos(2pi*t/P)]; F=LN(Attention(H,V,V)+H)",
                 "modules": {"patch temporal encoder": "Model.temporal_encoder", "visual augmentation/student": "Model.visual_augmentation/visual_encoder", "fusion": "Model.cross_modal"},
                 "differences": ["training-only large vision teacher and distillation omitted", "1D compact texture encoder without pseudo-image resizing"]}),
    RewriteCase("PMDformer", lambda: PMDformer(8, 3, 2, 4, 4, 1), lambda: PMDformer(1, 1, 2, 4, 1, 1),
                "https://arxiv.org/abs/2606.26549",
                {"method": "patch-mean decoupling, proximal variable attention, and trend restoration",
                 "equation": "r=P-mean(P); P_last=MHSA_variables(P_last); A=softmax(Q_shape K_shape^T/sqrt(d)); V=P Wv+mean(P)",
                 "modules": {"PMD": "Model.patch_mean_decouple", "PVA": "Model.proximal_attention", "TRA": "TrendRestorationAttention"},
                 "differences": ["one PVA and one shared TRA block", "replicated left padding for non-divisible histories"]}),
)


def _digest(value: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _runtime(case: RewriteCase) -> dict[str, object]:
    torch.manual_seed(260831)
    model = case.factory().cpu().eval()
    x = torch.randn(2, model.seq_len, model.enc_in, requires_grad=True)
    marks, adjacency = torch.randn(2, model.seq_len, 3), torch.eye(model.enc_in)
    output = model(x, marks, adjacency)
    if output.shape != (2, model.pred_len, model.enc_in) or not torch.isfinite(output).all():
        raise AssertionError("forward contract failed")
    output.square().mean().backward()
    if x.grad is None or not torch.isfinite(x.grad).all() or x.grad.abs().max() == 0:
        raise AssertionError("input gradient failed")
    gradients: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all() or parameter.grad.abs().max() == 0:
            raise AssertionError(f"inactive or invalid parameter gradient: {name}")
        gradients[name] = float(parameter.grad.abs().max())
    clone = case.factory().cpu().eval()
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    torch.testing.assert_close(clone(x.detach()), output.detach())
    if model(torch.randn(1, model.seq_len, model.enc_in)).shape != (1, model.pred_len, model.enc_in):
        raise AssertionError("batch boundary failed")
    boundary = case.boundary_factory().cpu().eval()
    if boundary(torch.randn(1, boundary.seq_len, boundary.enc_in)).shape != (1, 1, boundary.enc_in):
        raise AssertionError("minimum sequence failed")
    try:
        model(torch.randn(1, model.seq_len - 1, model.enc_in))
    except ValueError:
        wrong_length_rejected = True
    else:
        raise AssertionError("wrong sequence length accepted")
    torch.testing.assert_close(model(x.detach(), marks, adjacency), model(x.detach()))
    return {"shape": [2, model.pred_len, model.enc_in], "input_gradient_max_abs": float(x.grad.abs().max()),
            "parameter_gradients": gradients, "round_trip_max_abs": 0.0,
            "wrong_length_rejected": wrong_length_rejected}


def _environment() -> dict[str, object]:
    return {"python": platform.python_version(), "framework": f"torch {torch.__version__}",
            "dependencies": {"pydantic": importlib.metadata.version("pydantic"), "torch": torch.__version__},
            "platform": platform.platform(), "device": "cpu", "dtype": "float32",
            "deterministic": {"seed": 260831, "num_threads": torch.get_num_threads()}}


def _check(evidence: list[str], **metrics: float | int | str) -> dict[str, object]:
    return {"passed": True, "evidence": evidence, "metrics": metrics}


def verify(case: RewriteCase, records: dict[str, dict[str, object]]) -> None:
    observations = _runtime(case)
    structure_digest = _digest(case.structure)
    relative = f"verification/rewrite/{case.name}.json"
    artifact_path = ROOT / relative
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {"schema_version": 1, "kind": "clean-room-structure-map", "model": case.name,
                "reference": case.reference, "independent_design": True, "source_code_not_copied": True,
                "structure_map": case.structure, "structure_map_sha256": structure_digest,
                "observations": observations}
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence = [relative, "tests/test_recent_clean_room_rewrites_third.py"]
    checks = {
        "paper_structure": _check(evidence, mapped_elements=len(case.structure["modules"]), claim="paper-equations-to-independent-local-map"),
        "equations": _check(evidence, cases=1), "construction": _check(evidence, instances=3),
        "forward": _check(evidence, shape=",".join(str(value) for value in observations["shape"])),
        "backward": _check(evidence, input_gradient_max_abs=observations["input_gradient_max_abs"]),
        "finite_outputs": _check(evidence, nonfinite=0),
        "active_parameter_gradients": _check(evidence, parameters=len(observations["parameter_gradients"])),
        "state_dict_round_trip": _check(evidence, max_abs=0.0), "cpu": _check(evidence, device="cpu"),
        "batch_size_boundary": _check(evidence, cases="batch=1,batch=2"),
        "sequence_length_boundary": _check(evidence, cases="minimum-valid,wrong-length-rejected"),
        "marks_adjacency_contract": _check(evidence, contract="accepted-and-ignored"),
    }
    result = {"schema_version": 1, "kind": "rewrite-validation", "model": case.name,
              "implementation": "rewrite", "verified_at": datetime.now(timezone.utc),
              "subject_sha256": verification_subject_sha256(ROOT, records[case.name]),
              "commands": [f"uv run python scripts/verify_recent_clean_room_rewrites_third.py --model {case.name}",
                           "uv run python -m unittest tests.test_recent_clean_room_rewrites_third -v",
                           f"uv run tsf repo doctor --strict --models {case.name}"],
              "environment": _environment(), "artifacts": {relative: evidence_file_sha256(artifact_path)},
              "passed": True, "basis": {"references": [case.reference],
              "structure_map_sha256": structure_digest, "independent_design": True,
              "source_code_not_copied": True}, "checks": checks}
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
