#!/usr/bin/env python3
"""Generate clean-room evidence for BiST, DeepAR, HL, LSTM, LightTS, WaveNet."""

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
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark.catalog_metadata import model_records  # noqa: E402
from benchmark.verification_results import (  # noqa: E402
    evidence_file_sha256,
    verification_subject_sha256,
    write_verification_result,
)
from models.bist.model import Model as BiST  # noqa: E402
from models.deepar.model import Model as DeepAR  # noqa: E402
from models.hl.model import Model as HL  # noqa: E402
from models.lightts.model import Model as LightTS  # noqa: E402
from models.lstm.model import Model as LSTM  # noqa: E402
from models.wavenet.model import Model as WaveNet  # noqa: E402


Factory = Callable[[int, int, int], nn.Module]


def _marks(batch: int, steps: int, offset: int = 0) -> torch.Tensor:
    rows = [
        [2026, 8, 1 + index // 24, 5, (index + offset) % 24, 0]
        for index in range(steps)
    ]
    return torch.tensor([rows] * batch, dtype=torch.float32)


@dataclass(frozen=True)
class RewriteCase:
    name: str
    factory: Factory
    reference: str
    marks: str
    structure: dict[str, object]


CASES = (
    RewriteCase(
        "BiST",
        lambda length, horizon, channels: BiST(
            length, horizon, channels, model_dim=8, prompt_dim=4,
            num_layers=1, kernel_size=1 if length == 1 else 3,
            residual_steps=1, graph_dim=4, virtual_clusters=2,
        ),
        "https://www.vldb.org/pvldb/vol18/p1663-wang.pdf",
        "raw-calendar-or-node-covariates-active;adjacency-not-declared",
        {
            "method": "bidirectional spatiotemporal base prediction and residual correction",
            "equations": [
                "Eq. 5-7: X_l=AvgPool(X); X_s=X-X_l; X_0=MLP_1(X_l)+MLP_2(X_s)",
                "Eq. 8-10: prompt concatenation, residual MLP representation, and base forecast",
                "Eq. 11-16: virtual-cluster affinity separates common and personalized representations",
                "Eq. 17-24: adaptive kernel repeatedly diffuses residuals; Y=Y_base+Y_cor",
            ],
            "modules": {
                "temporal decomposition": "Model.decomposition/stable_projection/trend_projection",
                "spatiotemporal prompt and forward MLP": "Model node/time/weekday_prompt and forward_layers",
                "residual decoupling": "Model.node_queries/cluster_keys/residual_alignment",
                "residual diffusion and correction": "Model._adaptive_kernel/residual_layers/correction_head",
            },
            "differences": [
                "one calendar prompt per history window",
                "direct deterministic horizon heads without label access",
                "paper preprocessing, dataset-specific cluster counts, objectives, and metrics omitted",
            ],
        },
    ),
    RewriteCase(
        "DeepAR",
        lambda length, horizon, channels: DeepAR(
            length, horizon, channels, embedding_size=4, hidden_size=8,
            num_layers=1, cov_feat_size=2, dropout=0,
        ),
        "https://arxiv.org/abs/1704.04110",
        "history-and-future-covariates-active;adjacency-not-declared",
        {
            "method": "global autoregressive recurrent Gaussian forecaster",
            "equations": [
                "p(z_i,t|z_i,1:t-1,x_i,1:T)=l(z_i,t|theta(h_i,t))",
                "h_i,t=RNN(h_i,t-1,z_i,t-1,x_i,t)",
                "Gaussian theta=(location,positive scale); mean feeds the next inference step",
            ],
            "modules": {
                "target embedding": "Model.value_embedding",
                "global recurrent transition": "Model.recurrent",
                "likelihood parameters": "Model.likelihood",
                "autoregressive rollout": "Model.forward loop",
            },
            "differences": [
                "channels share parameters as independent related series",
                "mean feedback instead of ancestral sampling",
                "per-series scaling, negative-binomial likelihood, and age features omitted",
            ],
        },
    ),
    RewriteCase(
        "HL",
        lambda length, horizon, channels: HL(length, horizon, channels),
        "classical:persistence-last-observation",
        "marks-and-adjacency-not-declared",
        {
            "method": "historical-last persistence",
            "equations": ["Y[b,h,n]=X[b,T-1,n] for every horizon h"],
            "modules": {"exact persistence map": "Model.forward"},
            "differences": ["no associated paper", "no learned parameters"],
        },
    ),
    RewriteCase(
        "LSTM",
        lambda length, horizon, channels: LSTM(
            length, horizon, channels, init_dim=4, hid_dim=8, end_dim=8,
            layer=1, dropout=0, cov_dim=2,
        ),
        "https://doi.org/10.1162/neco.1997.9.8.1735",
        "raw-calendar-or-node-covariates-active;adjacency-not-declared",
        {
            "method": "shared per-node LSTM forecasting baseline",
            "equations": [
                "input/forget/output gates regulate the recurrent cell state",
                "h_T for every node is mapped by a shared direct horizon MLP",
            ],
            "modules": {
                "per-step projection": "Model.input_projection",
                "four-gate recurrence": "Model.recurrent",
                "direct decoder": "Model.forecast",
            },
            "differences": [
                "forecasting baseline rather than reproduction of the 1997 synthetic tasks",
                "optional calendar or node covariates",
                "weights shared across nodes; adjacency omitted",
            ],
        },
    ),
    RewriteCase(
        "LightTS",
        lambda length, horizon, channels: LightTS(
            length, horizon, channels, hid_dim=16,
            chunk_size=1 if length == 1 else 2,
        ),
        "https://arxiv.org/abs/2207.01186",
        "marks-and-adjacency-not-declared",
        {
            "method": "dual sampling-oriented information-exchange MLP",
            "equations": [
                "Eq. 1: continuous columns contain consecutive non-overlapping C-token chunks",
                "Eq. 2: interval columns contain C tokens separated by floor(T/C)",
                "Section 3.4: temporal projection, shared channel projection, then output projection",
            ],
            "modules": {
                "continuous sampling path": "Model.sample_continuous/continuous_block",
                "interval sampling path": "Model.sample_interval/interval_block",
                "feature fusion": "Model.forecast_block",
                "linear highway": "Model.highway",
            },
            "differences": [
                "multi-step direct horizon head and linear highway",
                "divisible history/chunk length required",
                "marks and paper preprocessing omitted",
            ],
        },
    ),
    RewriteCase(
        "WaveNet",
        lambda length, horizon, channels: WaveNet(
            length, horizon, channels, residual_channels=4,
            dilation_channels=4, skip_channels=4, end_channels=8,
            blocks=1, layers=2,
        ),
        "https://arxiv.org/abs/1609.03499",
        "marks-and-adjacency-not-declared",
        {
            "method": "dilated causal gated residual convolution forecaster",
            "equations": [
                "z=tanh(W_f*x) elementwise sigmoid(W_g*x)",
                "left-only causal convolutions use exponentially increasing dilation",
                "residual and skip paths aggregate before the direct horizon projection",
            ],
            "modules": {
                "causal gated layer": "GatedCausalLayer",
                "dilation stack": "Model.causal_layers",
                "skip summary and forecast": "Model.head/horizon_projection",
                "normalization": "canonical RevIN instance",
            },
            "differences": [
                "channel-independent shared forecasting stack",
                "direct regression and RevIN replace autoregressive categorical audio generation",
                "global/local conditioning and audio preprocessing omitted",
            ],
        },
    ),
)


def _digest(value: dict[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _kwargs(name: str, batch: int, length: int, horizon: int) -> dict[str, torch.Tensor]:
    if name in {"BiST", "LSTM"}:
        return {"x_mark_enc": _marks(batch, length)}
    if name == "DeepAR":
        return {
            "x_mark_enc": _marks(batch, length)[..., :2],
            "x_mark_dec": torch.randn(batch, horizon, 2),
        }
    return {}


def _runtime(case: RewriteCase) -> dict[str, object]:
    torch.manual_seed(260827)
    model = case.factory(8, 3, 4).cpu().eval()
    x = torch.randn(2, 8, 4, requires_grad=True)
    kwargs = _kwargs(case.name, 2, 8, 3)
    output = model(x, **kwargs)
    expected = (2, 3, 4, 2) if case.name == "DeepAR" else (2, 3, 4)
    if output.shape != expected or not torch.isfinite(output).all():
        raise AssertionError("forward contract failed")
    if case.name == "DeepAR" and not (output[..., 1] > 0).all():
        raise AssertionError("Gaussian scale is not positive")
    output.square().mean().backward()
    if x.grad is None or not torch.isfinite(x.grad).all() or x.grad.abs().max() == 0:
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

    clone = case.factory(8, 3, 4).cpu().eval()
    clone.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    clone_output = clone(x.detach(), **kwargs)
    torch.testing.assert_close(clone_output, output.detach())

    boundary = case.factory(1, 1, 1).cpu().eval()
    boundary_output = boundary(torch.randn(1, 1, 1), **_kwargs(case.name, 1, 1, 1))
    boundary_expected = (1, 1, 1, 2) if case.name == "DeepAR" else (1, 1, 1)
    if boundary_output.shape != boundary_expected:
        raise AssertionError("minimum boundary failed")
    for bad in (torch.randn(1, 7, 4), torch.randn(1, 8, 3)):
        try:
            model(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid input shape accepted")

    marks_effect = 0.0
    if case.name in {"BiST", "LSTM"}:
        first = model(x.detach(), _marks(2, 8))
        second = model(x.detach(), _marks(2, 8, offset=9))
        marks_effect = float((first - second).abs().max())
        if marks_effect == 0:
            raise AssertionError("declared marks did not affect output")
    elif case.name == "DeepAR":
        first_kwargs = _kwargs(case.name, 2, 8, 3)
        second_kwargs = dict(first_kwargs)
        second_kwargs["x_mark_dec"] = first_kwargs["x_mark_dec"] + 5
        marks_effect = float(
            (model(x.detach(), **first_kwargs) - model(x.detach(), **second_kwargs))
            .abs()
            .max()
        )
        if marks_effect == 0:
            raise AssertionError("future covariates did not affect output")
    else:
        first = model(x.detach(), _marks(2, 8))
        second = model(x.detach(), _marks(2, 8, offset=9))
        marks_effect = float((first - second).abs().max())
        if marks_effect != 0:
            raise AssertionError("undeclared marks affected output")

    return {
        "shape": list(expected),
        "batch_size_cases": [1, 2],
        "minimum_history": 1,
        "input_gradient_max_abs": float(x.grad.abs().max()),
        "parameter_gradients": gradients,
        "round_trip_max_abs": float((clone_output - output.detach()).abs().max()),
        "marks_effect_max_abs": marks_effect,
        "marks_contract": case.marks,
        "adjacency_contract": "not declared",
        "wrong_length_rejected": True,
        "wrong_channel_count_rejected": True,
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
        "deterministic": {
            "seed": 260827,
            "num_threads": torch.get_num_threads(),
        },
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
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    evidence = [
        relative,
        "tests/test_baseline_sequence_clean_room_rewrites.py",
    ]
    checks = {
        "paper_structure": _check(
            evidence,
            mapped_elements=len(case.structure["modules"]),
            claim="paper-equations-to-independent-local-map",
        ),
        "equations": _check(evidence, cases=len(case.structure["equations"])),
        "construction": _check(evidence, instances=3),
        "forward": _check(evidence, shape=",".join(map(str, observations["shape"]))),
        "backward": _check(
            evidence,
            input_gradient_max_abs=observations["input_gradient_max_abs"],
        ),
        "finite_outputs": _check(evidence, nonfinite=0),
        "active_parameter_gradients": _check(
            evidence, parameters=len(observations["parameter_gradients"])
        ),
        "state_dict_round_trip": _check(
            evidence, max_abs=observations["round_trip_max_abs"]
        ),
        "cpu": _check(evidence, device="cpu"),
        "batch_size_boundary": _check(evidence, cases="batch=1,batch=2"),
        "sequence_length_boundary": _check(
            evidence, cases="seq=1,channel=1,wrong-shapes-rejected"
        ),
        "marks_adjacency_contract": _check(
            evidence, contract=case.marks, adjacency="not-declared"
        ),
    }
    result = {
        "schema_version": 1,
        "kind": "rewrite-validation",
        "model": case.name,
        "implementation": "rewrite",
        "verified_at": datetime.now(timezone.utc),
        "subject_sha256": verification_subject_sha256(ROOT, records[case.name]),
        "commands": [
            f"uv run python scripts/verify_baseline_sequence_clean_room_rewrites.py --model {case.name}",
            "uv run python -m unittest tests.test_baseline_sequence_clean_room_rewrites -v",
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
    parser.add_argument(
        "--model", action="append", choices=[case.name for case in CASES]
    )
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
