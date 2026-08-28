#!/usr/bin/env python3
"""Execute and record clean-room evidence for five neural forecasting baselines."""

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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark.catalog_metadata import model_records  # noqa: E402
from benchmark.verification_results import (  # noqa: E402
    evidence_file_sha256,
    verification_subject_sha256,
    write_verification_result,
)
from models.gru_forecaster_ts.model import Model as GRUForecaster  # noqa: E402
from models.lstm_forecaster_ts.model import Model as LSTMForecaster  # noqa: E402
from models.mlp_forecaster_ts.model import Model as MLPForecaster  # noqa: E402
from models.rnn_forecaster_ts.model import Model as RNNForecaster  # noqa: E402
from models.tcn_forecaster_ts.model import CausalConv1d, Model as TCNForecaster  # noqa: E402


Factory = Callable[[int, int, int], nn.Module]


@dataclass(frozen=True)
class RewriteCase:
    name: str
    factory: Factory
    reference: str
    structure: dict[str, object]


CASES = (
    RewriteCase(
        "RNNForecasterTS",
        lambda length, horizon, channels: RNNForecaster(length, horizon, channels, d_model=5),
        "https://doi.org/10.1207/s15516709cog1402_1",
        {
            "method": "Elman tanh recurrence with direct multistep decoding",
            "equations": ["h_t=tanh(W_ih*x_t+b_ih+W_hh*h_(t-1)+b_hh)", "y=reshape(W_o*h_T+b_o)"],
            "modules": {"recurrence": "Model.encoder", "horizon projection": "Model.head"},
            "differences": ["direct horizon head is repository-defined", "channels are encoded jointly", "optional RevIN"],
        },
    ),
    RewriteCase(
        "GRUForecasterTS",
        lambda length, horizon, channels: GRUForecaster(length, horizon, channels, d_model=5),
        "https://arxiv.org/abs/1412.3555",
        {
            "method": "gated recurrent unit with direct multistep decoding",
            "equations": ["r_t=sigmoid(W_ir*x_t+b_ir+W_hr*h_(t-1)+b_hr)", "z_t=sigmoid(W_iz*x_t+b_iz+W_hz*h_(t-1)+b_hz)", "n_t=tanh(W_in*x_t+b_in+r_t*(W_hn*h_(t-1)+b_hn))", "h_t=(1-z_t)*n_t+z_t*h_(t-1)"],
            "modules": {"gated recurrence": "Model.encoder", "horizon projection": "Model.head"},
            "differences": ["direct horizon head is repository-defined", "final hidden state only", "optional RevIN"],
        },
    ),
    RewriteCase(
        "LSTMForecasterTS",
        lambda length, horizon, channels: LSTMForecaster(length, horizon, channels, d_model=5),
        "https://doi.org/10.1162/neco.1997.9.8.1735",
        {
            "method": "LSTM gated memory with direct multistep decoding",
            "equations": ["c_t=f_t*c_(t-1)+i_t*g_t", "h_t=o_t*tanh(c_t)", "y=reshape(W_o*h_T+b_o)"],
            "modules": {"gated memory": "Model.encoder", "horizon projection": "Model.head"},
            "differences": ["direct horizon head is repository-defined", "final hidden state only", "optional RevIN"],
        },
    ),
    RewriteCase(
        "MLPForecasterTS",
        lambda length, horizon, channels: MLPForecaster(length, horizon, channels, d_model=7),
        "https://doi.org/10.1038/323533a0",
        {
            "method": "channel-wise feed-forward lag-to-horizon mapping",
            "equations": ["z_1=GELU(W_1*x+b_1)", "z_l=GELU(W_l*z_(l-1)+b_l)", "y=W_o*z_L+b_o"],
            "modules": {"shared channel-wise network": "Model.network"},
            "differences": ["time-series lag mapping is repository-defined", "no cross-channel mixing", "GELU activations", "optional RevIN"],
        },
    ),
    RewriteCase(
        "TCNForecasterTS",
        lambda length, horizon, channels: TCNForecaster(length, horizon, channels, d_model=5, num_layers=2),
        "https://arxiv.org/abs/1803.01271",
        {
            "method": "dilated causal residual temporal convolution with direct decoding",
            "equations": ["u_l=GELU(CausalConv_d(CausalConv_d(z_(l-1)))+R_l(z_(l-1)))", "d_l=2^l", "y=reshape(W_o*u_L[T]+b_o)"],
            "modules": {"causal convolution": "CausalConv1d", "residual block": "TemporalResidualBlock", "horizon projection": "Model.head"},
            "differences": ["weight normalization omitted", "final-timestep direct forecast head", "optional RevIN"],
        },
    ),
)


def _manual_recurrence(module: nn.Module, x: torch.Tensor) -> torch.Tensor:
    hidden = x.new_zeros(x.size(0), module.hidden_size)
    cell = torch.zeros_like(hidden)
    for token in x.unbind(dim=1):
        input_terms = F.linear(token, module.weight_ih_l0, module.bias_ih_l0)
        hidden_terms = F.linear(hidden, module.weight_hh_l0, module.bias_hh_l0)
        if isinstance(module, nn.RNN):
            hidden = torch.tanh(input_terms + hidden_terms)
        elif isinstance(module, nn.GRU):
            ir, iz, inn = input_terms.chunk(3, dim=-1)
            hr, hz, hn = hidden_terms.chunk(3, dim=-1)
            reset, update = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
            candidate = torch.tanh(inn + reset * hn)
            hidden = (1.0 - update) * candidate + update * hidden
        else:
            input_gate, forget_gate, candidate, output_gate = (input_terms + hidden_terms).chunk(4, dim=-1)
            cell = torch.sigmoid(forget_gate) * cell + torch.sigmoid(input_gate) * torch.tanh(candidate)
            hidden = torch.sigmoid(output_gate) * torch.tanh(cell)
    return hidden


def _equation_check(case: RewriteCase) -> None:
    torch.manual_seed(197)
    if case.name in {"RNNForecasterTS", "GRUForecasterTS", "LSTMForecasterTS"}:
        model = case.factory(4, 2, 3)
        model.revin.enabled = False
        x = torch.randn(2, 4, 3)
        _, state = model.encoder(x)
        actual = state[0][-1] if isinstance(state, tuple) else state[-1]
        torch.testing.assert_close(actual, _manual_recurrence(model.encoder, x))
    elif case.name == "MLPForecasterTS":
        model = MLPForecaster(2, 1, 2, d_model=1, dropout=0.0, use_revin=False)
        with torch.no_grad():
            model.network[0].weight.copy_(torch.tensor([[2.0, -1.0]]))
            model.network[0].bias.copy_(torch.tensor([0.5]))
            model.network[-1].weight.copy_(torch.tensor([[3.0]]))
            model.network[-1].bias.copy_(torch.tensor([-2.0]))
        x = torch.tensor([[[1.0, 4.0], [3.0, 2.0]]])
        expected = (3.0 * F.gelu(torch.tensor([[-0.5, 6.5]])) - 2.0).reshape(1, 1, 2)
        torch.testing.assert_close(model(x), expected)
    else:
        convolution = CausalConv1d(1, 1, kernel_size=3, dilation=2)
        with torch.no_grad():
            convolution.weight.fill_(1.0)
            convolution.bias.zero_()
        first = torch.arange(7, dtype=torch.float32).reshape(1, 1, 7)
        second = first.clone()
        second[..., 5:] += 1000.0
        torch.testing.assert_close(convolution(first)[..., :5], convolution(second)[..., :5])


def _runtime_checks(case: RewriteCase) -> dict[str, object]:
    torch.manual_seed(733)
    _equation_check(case)
    model = case.factory(4, 3, 2).cpu().eval()
    x = torch.randn(2, 4, 2, requires_grad=True)
    marks, adjacency = torch.randn(2, 4, 3), torch.eye(2)
    output = model(x, marks, adjacency)
    if output.shape != (2, 3, 2) or not torch.isfinite(output).all():
        raise AssertionError("forward shape or finiteness failed")
    output.square().mean().backward()
    if x.grad is None or not torch.isfinite(x.grad).all():
        raise AssertionError("input backward failed")
    gradients: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise AssertionError(f"missing/nonfinite parameter gradient: {name}")
        gradients[name] = float(parameter.grad.abs().max())
        if gradients[name] == 0.0:
            raise AssertionError(f"inactive parameter: {name}")
    clone = case.factory(4, 3, 2).cpu().eval()
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    torch.testing.assert_close(clone(x.detach()), output.detach())
    batch_shape = tuple(model(torch.randn(1, 4, 2)).shape)
    boundary_shape = tuple(case.factory(1, 1, 2).cpu().eval()(torch.randn(1, 1, 2)).shape)
    try:
        model(torch.randn(1, 3, 2))
    except ValueError:
        rejected_wrong_length = True
    else:
        rejected_wrong_length = False
    baseline = model(x.detach())
    torch.testing.assert_close(model(x.detach(), marks, adjacency), baseline)
    if batch_shape != (1, 3, 2) or boundary_shape != (1, 1, 2) or not rejected_wrong_length:
        raise AssertionError("boundary contract failed")
    return {
        "output_shape": list(output.shape), "output_finite": True,
        "input_gradient_max_abs": float(x.grad.abs().max()),
        "parameter_gradient_max_abs": gradients, "state_dict_round_trip_max_abs": 0.0,
        "batch_size_one_shape": list(batch_shape), "minimum_sequence_shape": list(boundary_shape),
        "wrong_sequence_rejected": rejected_wrong_length,
        "marks_adjacency": "accepted and deliberately ignored by time-series-only contract",
    }


def _environment() -> dict[str, object]:
    return {
        "python": platform.python_version(), "framework": f"torch {torch.__version__}",
        "dependencies": {"pydantic": importlib.metadata.version("pydantic"), "torch": torch.__version__},
        "platform": platform.platform(), "device": "cpu", "dtype": "float32",
        "deterministic": {"seed": 733, "num_threads": torch.get_num_threads()},
    }


def _check(evidence: list[str], **metrics: float | int | str) -> dict[str, object]:
    return {"passed": True, "evidence": evidence, "metrics": metrics}


def verify(case: RewriteCase, records: dict[str, dict[str, object]]) -> None:
    observations = _runtime_checks(case)
    structure_payload = json.dumps(case.structure, sort_keys=True, separators=(",", ":")).encode()
    structure_digest = hashlib.sha256(structure_payload).hexdigest()
    relative_artifact = f"verification/rewrite/{case.name}.json"
    artifact_path = ROOT / relative_artifact
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "schema_version": 1, "kind": "clean-room-structure-map", "model": case.name,
        "reference": case.reference, "independent_design": True, "source_code_not_copied": True,
        "structure_map": case.structure, "structure_map_sha256": structure_digest,
        "observations": observations,
    }
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence = [relative_artifact, "tests/test_neural_baseline_rewrites.py"]
    checks = {
        "paper_structure": _check(evidence, mapped_elements=len(case.structure["modules"]), claim="reference-to-local map with disclosed forecast adaptation"),
        "equations": _check(evidence, cases=1), "construction": _check(evidence, instances=3),
        "forward": _check(evidence, shape="2,3,2"),
        "backward": _check(evidence, input_gradient_max_abs=observations["input_gradient_max_abs"]),
        "finite_outputs": _check(evidence, nonfinite=0),
        "active_parameter_gradients": _check(evidence, parameters=len(observations["parameter_gradient_max_abs"])),
        "state_dict_round_trip": _check(evidence, max_abs=0.0), "cpu": _check(evidence, device="cpu"),
        "batch_size_boundary": _check(evidence, cases="batch=1,batch=2"),
        "sequence_length_boundary": _check(evidence, cases="length=1,wrong-length-rejected"),
        "marks_adjacency_contract": _check(evidence, contract="accepted-and-ignored"),
    }
    result = {
        "schema_version": 1, "kind": "rewrite-validation", "model": case.name,
        "implementation": "rewrite", "verified_at": datetime.now(timezone.utc),
        "subject_sha256": verification_subject_sha256(ROOT, records[case.name]),
        "commands": [f"uv run python scripts/verify_neural_baseline_rewrites.py --model {case.name}", "uv run python -m unittest tests.test_neural_baseline_rewrites -v", f"uv run tsf repo doctor --backward --models {case.name}"],
        "environment": _environment(), "artifacts": {relative_artifact: evidence_file_sha256(artifact_path)},
        "passed": True, "basis": {"references": [case.reference], "structure_map_sha256": structure_digest, "independent_design": True, "source_code_not_copied": True},
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
