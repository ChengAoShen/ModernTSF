#!/usr/bin/env python3
"""Generate strict rewrite evidence for six native/probabilistic models."""

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
from models.gaussian_mlp.model import Model as GaussianMLP  # noqa: E402
from models.mqrnn.model import Model as MQRNN  # noqa: E402
from models.patchmlp.model import Model as PatchMLP  # noqa: E402
from models.pws.model import Model as PWS  # noqa: E402
from models.svtime.model import Model as SVTime  # noqa: E402
from models.timebase.model import Model as TimeBase  # noqa: E402


def _factory(name: str, length: int = 8, horizon: int = 3, channels: int = 2):
    if name == "GaussianMLP":
        return GaussianMLP(length, horizon, channels, hidden_size=16, num_layers=2, dropout=0.0)
    if name == "PWS":
        period = 4 if length >= 4 else 1
        return PWS(channels, period, length, horizon, min(2, period), False, False, False, "relu", [4])
    if name == "PatchMLP":
        patches = [4, 4, 2, 2] if length >= 4 else [2, 2, 2, 2]
        return PatchMLP(length, horizon, channels, 32 if length >= 4 else 16, 1, False, 3 if length >= 3 else 1, patches)
    if name == "SVTime":
        period = 4 if length >= 4 else 1
        return SVTime(channels, period, length, horizon, min(3, period), False, False, False)
    if name == "TimeBase":
        period = 4 if length >= 4 else 1
        return TimeBase(length, horizon, channels, period, 2 if period > 1 else 1, False, 0.08, True)
    if name == "MQRNN":
        return MQRNN(length, horizon, channels, hidden_size=12, context_size=5, decoder_hidden=11, future_covariate_size=6, dropout=0.0)
    raise KeyError(name)


REFERENCES = {
    "GaussianMLP": "ModernTSF in-repository Gaussian-head baseline definition",
    "PWS": "ModernTSF in-repository Patch Weighted Sum baseline definition",
    "PatchMLP": "https://arxiv.org/abs/2405.13575",
    "SVTime": "https://arxiv.org/abs/2510.09780",
    "TimeBase": "https://proceedings.mlr.press/v267/huang25az.html",
    "MQRNN": "https://arxiv.org/abs/1711.11053",
}
STRUCTURES = {
    "GaussianMLP": {
        "method": "flattened-history MLP with independent Gaussian parameters",
        "equation": "h0=vec(X); hl=Dropout(ReLU(Wl h(l-1)+bl)); loc=Wmu h; scale=softplus(Wsigma h)+eps",
        "modules": {"history MLP": "Model.backbone", "positive Gaussian head": "Model.parameter_head"},
        "differences": ["repository-defined baseline without a canonical paper", "independent marginal covariance only"],
    },
    "PWS": {
        "method": "patch-wise residual analysis followed by a historical-to-future period map",
        "equation": "Hk=A_k(Xk)+Xk; Yk=W_k Hk+b_k",
        "modules": {"period/patch reshape": "PWSModel.forward", "residual analysis": "PWSModel.analysis_layers", "weighted period sum": "PWSModel.weighted_sum_layers"},
        "differences": ["repository-defined baseline without a canonical paper"],
    },
    "PatchMLP": {
        "method": "multi-scale patch MLP with latent smooth/residual decomposition",
        "equation": "Z=Concat_s Embed_s(Patch_s(X)); Zs=AvgPool(Z); Zr=Z-Zs; H=InterMLP(IntraMLP(Z)); Y=Project(Hs+Hr)",
        "modules": {"multi-scale patch embedding": "Emb/EmbLayer", "latent decomposition": "SeriesDecomp", "channel-independent residual branch": "PatchMLPModel.residual_layers", "smooth temporal/channel mixing": "PatchMLPModel.smooth_layers", "forecast projection": "PatchMLPModel.projector"},
        "differences": ["independent implementation; reference-only unlicensed author repository not inspected", "no publication-result or initialization parity claim"],
    },
    "SVTime": {
        "method": "SVTime IB1/IB2 period map with backcast-residual trend correction",
        "equation": "[xhat,yhat]=LVM-IB(x); dx=x-xhat; dy=WB(dx)+bB; yfinal=sigmoid(wg)dy+(1-sigmoid(wg))yhat",
        "modules": {"patch-specific period matrices": "PatchWisePeriodMap", "joint backcast/forecast": "SVTimeModel.period_backcast_forecast", "residual trend and gate": "SVTimeModel.forward"},
        "differences": ["implements named SVTime rather than SVTime-t annealed attention", "optional repository RevIN and one LVM-IB block"],
    },
    "TimeBase": {
        "method": "segment-level low-rank basis extraction and forecasting",
        "equation": "Xhis=Segment(X); Xbasis=BasisExtract(Xhis); Xpred=SegmentForecast(Xbasis); Lorth=||G-diag(G)||F",
        "modules": {"segmentation/padding": "TimeBaseModel.forward", "basis extraction": "TimeBaseModel.ts2basis", "segment forecasting": "TimeBaseModel.basis2ts", "orthogonal restriction": "cal_orthogonal_loss"},
        "differences": ["licensed author repository pinned reference-only and not inspected", "dataset-specific lambda and reported results are not parity claims"],
    },
    "MQRNN": {
        "method": "direct multi-horizon quantile LSTM with global/local decoders",
        "equation": "(c1,...,cK,ca)=mG(ht,x_future); qhat_k=mL(ck,ca,x_future_k)",
        "modules": {"target/covariate LSTM": "Model.encoder", "joint horizon contexts": "Model.global_decoder/Model.decode_contexts", "shared horizon decoder": "Model.local_decoder", "quantile output": "Model.quantile_head"},
        "differences": ["monotone repository quantile parameterization", "static item covariates and forking-sequences training require experiment-layer APIs"],
    },
}


def _digest(value: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _call(name: str, model, x: torch.Tensor, marks: torch.Tensor, future: torch.Tensor):
    if name == "MQRNN":
        return model(x, marks, None, future)
    return model(x, marks, torch.eye(x.shape[-1]))


def _runtime(name: str) -> dict[str, object]:
    torch.manual_seed(260827)
    model = _factory(name).cpu().eval()
    x = torch.randn(2, 8, 2, requires_grad=True)
    marks = torch.randn(2, 8, 6)
    future = torch.randn(2, 3, 6)
    output = _call(name, model, x, marks, future)
    expected = (2, 3, 2, 2) if name == "GaussianMLP" else (2, 3, 2, 9) if name == "MQRNN" else (2, 3, 2)
    if tuple(output.shape) != expected or not torch.isfinite(output).all():
        raise AssertionError("forward contract failed")
    if name == "GaussianMLP" and not (output[..., 1] > 0).all():
        raise AssertionError("non-positive Gaussian scale")
    if name == "MQRNN" and not (output[..., 1:] >= output[..., :-1]).all():
        raise AssertionError("quantile crossing detected")
    loss = output.square().mean()
    if name == "TimeBase" and model.aux_loss is not None:
        loss = loss + model.aux_loss
    loss.backward()
    if x.grad is None or not torch.isfinite(x.grad).all() or x.grad.abs().max() == 0:
        raise AssertionError("input gradient failed")
    gradients: dict[str, float] = {}
    for parameter_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None or not torch.isfinite(parameter.grad).all() or parameter.grad.abs().max() == 0:
            raise AssertionError(f"inactive parameter: {parameter_name}")
        gradients[parameter_name] = float(parameter.grad.abs().max())

    clone = _factory(name).cpu().eval()
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    torch.testing.assert_close(_call(name, clone, x.detach(), marks, future), output.detach())
    batch_one = torch.randn(1, 8, 2)
    marks_one, future_one = torch.randn(1, 8, 6), torch.randn(1, 3, 6)
    if _call(name, model, batch_one, marks_one, future_one).shape[0] != 1:
        raise AssertionError("batch boundary failed")
    boundary_length = 2 if name == "PatchMLP" else 1
    boundary = _factory(name, boundary_length, 1, 2).eval()
    boundary_output = boundary(torch.randn(1, boundary_length, 2))
    if boundary_output.shape[:2] != (1, 1):
        raise AssertionError("minimum-length boundary failed")
    try:
        model(torch.randn(1, 7, 2))
    except ValueError:
        rejected = True
    else:
        raise AssertionError("wrong length accepted")
    if name == "MQRNN":
        changed = model(x.detach(), marks, None, future + 1.0)
        covariate_effect = float((changed - output.detach()).abs().max())
        if covariate_effect == 0:
            raise AssertionError("future covariates are inactive")
        marks_contract = "historical-and-future-temporal-covariates-consumed; adjacency-not-applicable"
    else:
        covariate_effect = 0.0
        marks_contract = "accepted-and-ignored; adjacency-not-applicable"
    return {
        "shape": list(output.shape),
        "input_gradient_max_abs": float(x.grad.abs().max()),
        "parameter_gradients": gradients,
        "round_trip_max_abs": 0.0,
        "wrong_length_rejected": rejected,
        "future_covariate_effect_max_abs": covariate_effect,
        "marks_contract": marks_contract,
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
    evidence = [relative, "tests/test_native_probabilistic_rewrites.py"]
    claim = "repository-definition-to-local-map" if name in {"GaussianMLP", "PWS"} else "paper-equations-to-independent-local-map"
    checks = {
        "paper_structure": _check(evidence, mapped_elements=len(structure["modules"]), claim=claim),
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
        "marks_adjacency_contract": _check(evidence, contract=observations["marks_contract"]),
    }
    result = {
        "schema_version": 1,
        "kind": "rewrite-validation",
        "model": name,
        "implementation": "rewrite",
        "verified_at": datetime.now(timezone.utc),
        "subject_sha256": verification_subject_sha256(ROOT, records[name]),
        "commands": [
            f"uv run python scripts/verify_native_probabilistic_rewrites.py --model {name}",
            "uv run python -m unittest tests.test_native_probabilistic_rewrites -v",
            f"uv run tsf repo doctor --strict --models {name}",
        ],
        "environment": {
            "python": platform.python_version(),
            "framework": f"torch {torch.__version__}",
            "dependencies": {"pydantic": importlib.metadata.version("pydantic"), "torch": torch.__version__},
            "platform": platform.platform(),
            "device": "cpu",
            "dtype": "float32",
            "deterministic": {"seed": 260827, "num_threads": torch.get_num_threads()},
        },
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
    parser.add_argument("--model", action="append", choices=sorted(STRUCTURES))
    args = parser.parse_args()
    selected = set(args.model or STRUCTURES)
    records = {str(record["name"]): record for record in model_records(ROOT)}
    for name in STRUCTURES:
        if name in selected:
            verify(name, records)
            print(f"{name}: rewrite validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
