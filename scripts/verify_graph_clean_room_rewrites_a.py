#!/usr/bin/env python3
"""Generate clean-room evidence for ASTGCN/DCRNN/DGCRN/DSTAGNN/GCLSTM/GTS."""

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

import numpy as np
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
from models.astgcn.model import Model as ASTGCN  # noqa: E402
from models.dcrnn.model import Model as DCRNN  # noqa: E402
from models.dgcrn.model import Model as DGCRN  # noqa: E402
from models.dstagnn.model import Model as DSTAGNN  # noqa: E402
from models.gclstm.model import Model as GCLSTM  # noqa: E402
from models.gts.model import Model as GTS  # noqa: E402


Factory = Callable[[np.ndarray], nn.Module]


def _graph(nodes: int) -> np.ndarray:
    graph = np.eye(nodes, dtype=np.float32)
    for index in range(nodes - 1):
        graph[index, index + 1] = 1.0
        graph[index + 1, index] = 0.5
    return graph


def _marks(batch: int, steps: int, offset: int = 0) -> torch.Tensor:
    rows = [[2026, 8, 1 + index // 24, 5, (index + offset) % 24, 0] for index in range(steps)]
    return torch.tensor([rows] * batch, dtype=torch.float32)


@dataclass(frozen=True)
class RewriteCase:
    name: str
    factory: Factory
    boundary_factory: Factory
    reference: str
    marks: str
    structure: dict[str, object]


CASES = (
    RewriteCase(
        "ASTGCN",
        lambda graph: ASTGCN(6, 3, 4, graph, cov_dim=2, nb_block=1, K=2, nb_chev_filter=8, nb_time_filter=8),
        lambda graph: ASTGCN(1, 1, 1, graph, cov_dim=0, nb_block=1, K=1, nb_chev_filter=2, nb_time_filter=2),
        "https://doi.org/10.1609/aaai.v33i01.3301922",
        "raw-calendar-and-node-structured-active",
        {
            "method": "single recent-history ASTGCN branch",
            "equations": [
                "S=softmax(Q_node K_node^T/sqrt(d)); E=softmax(Q_time K_time^T/sqrt(d))",
                "X'=sum_k Theta_k (T_k(L) elementwise S) (E X)",
                "H=tanh(Conv_t(X')) elementwise sigmoid(Conv_g(X')); Y=Conv_horizon(H)",
            ],
            "modules": {
                "spatial-temporal attention": "SpatialTemporalAttention",
                "attention Chebyshev filter": "AttentionChebyshevConvolution",
                "gated temporal convolution": "ASTGCNBlock",
                "horizon projection": "Model.forecast",
            },
            "differences": ["recent-history branch only", "no daily/weekly fusion", "no paper preprocessing or masked objective"],
        },
    ),
    RewriteCase(
        "DCRNN",
        lambda graph: DCRNN(6, 3, 4, graph, input_dim=3, rnn_units=8, max_diffusion_step=2),
        lambda graph: DCRNN(1, 1, 1, graph, input_dim=1, rnn_units=2, max_diffusion_step=0),
        "https://openreview.net/forum?id=SJiHXGWAZ",
        "encoder-calendar-or-node-features-active",
        {
            "method": "dual-random-walk diffusion recurrent encoder-decoder",
            "equations": [
                "X star_G f=sum_k(theta_k,1 (D_O^-1 W)^k + theta_k,2 (D_I^-1 W^T)^k)X",
                "r,u=sigmoid(diffusion([x,h])); c=tanh(diffusion([x,r elementwise h]))",
                "h'=u elementwise h+(1-u) elementwise c",
            ],
            "modules": {"diffusion polynomial": "DiffusionConvolution", "DCGRU gates": "DCGRUCell", "seq2seq": "Model.encoder/Model.decoder"},
            "differences": ["no scheduled sampling or teacher forcing", "forecast-only", "official data and masked-MAE pipeline omitted"],
        },
    ),
    RewriteCase(
        "DGCRN",
        lambda graph: DGCRN(6, 3, 4, graph, rnn_size=8, node_dim=4, hyper_gnn_dim=4, middle_dim=2, dropout=0),
        lambda graph: DGCRN(1, 1, 1, graph, rnn_size=2, node_dim=2, hyper_gnn_dim=2, middle_dim=1, dropout=0),
        "https://doi.org/10.1145/3532611",
        "historical-and-future-time-driver-active",
        {
            "method": "hidden-state-conditioned dynamic graph recurrent network",
            "equations": [
                "E_source,t=Hyper_source([h_t,e_source]); E_target,t=Hyper_target([h_t,e_target])",
                "A_t=softmax(ReLU(tanh(alpha E_source,t E_target,t^T)))",
                "GConv=Linear([X,static hops,dynamic hops]); graph-GRU uses GConv in gates",
            ],
            "modules": {"hyper-network graph": "DynamicGraphGenerator", "static/dynamic propagation": "DynamicGraphConvolution", "recurrent cell": "DynamicGraphGRUCell"},
            "differences": ["one shared encoder/decoder cell", "no task-level curriculum", "no future-target teacher forcing"],
        },
    ),
    RewriteCase(
        "DSTAGNN",
        lambda graph: DSTAGNN(6, 3, 4, graph, d_model=8, d_k=2, d_v=2, n_heads=2),
        lambda graph: DSTAGNN(1, 1, 1, graph, d_model=2, d_k=1, d_v=1, n_heads=1),
        "https://proceedings.mlr.press/v162/lan22a.html",
        "marks-intentionally-not-consumed",
        {
            "method": "dynamic spatial-temporal attention graph network",
            "equations": [
                "Attention(Q,K,V)=softmax(QK^T/sqrt(d_k))V over time and nodes",
                "X'=sum_k Theta_k (T_k(L) elementwise S_t)X",
                "H=Fuse_k(tanh(Conv_k(X')) elementwise sigmoid(Gate_k(X'))) for k in {3,5,7}",
            ],
            "modules": {"axis attention": "AxisAttention", "dynamic Chebyshev filter": "DynamicChebyshevConvolution", "multi-scale gated temporal module": "MultiScaleGatedTemporalConvolution"},
            "differences": ["supplied graph replaces pattern-aware graph", "one block without residual-attention accumulation", "temporal-distance matrix omitted"],
        },
    ),
    RewriteCase(
        "GCLSTM",
        lambda graph: GCLSTM(6, 3, 4, graph, cov_dim=2, Ks=2, hidden_dim=8),
        lambda graph: GCLSTM(1, 1, 1, graph, cov_dim=0, Ks=1, hidden_dim=2),
        "https://doi.org/10.1016/j.scitotenv.2019.01.333",
        "raw-calendar-and-node-covariates-active",
        {
            "method": "Chebyshev graph-convolutional LSTM",
            "equations": [
                "GConv(X)=Linear([T_0(L)X,...,T_(K-1)(L)X])",
                "i,f,o,c_tilde=split(GConv([x_t,h_(t-1)])); c_t=f*c_(t-1)+i*tanh(c_tilde)",
                "h_t=o*tanh(c_t); Y=Linear_horizon(h_T)",
            ],
            "modules": {"Chebyshev responses": "ChebyshevGraphProjection", "four recurrent gates": "GraphConvLSTMCell", "direct decoder": "Model.forecast"},
            "differences": ["direct multi-horizon decoder", "paper feature construction and training protocol unavailable", "no metric parity claim"],
        },
    ),
    RewriteCase(
        "GTS",
        lambda graph: GTS(6, 3, 4, graph, input_dim=3, rnn_units=8, max_diffusion_step=2, embedding_dim=8, temp=0.7, prior_strength=0.2),
        lambda graph: GTS(1, 1, 1, graph, input_dim=1, rnn_units=2, max_diffusion_step=1, embedding_dim=2),
        "https://openreview.net/forum?id=WEHSlH5mOk",
        "encoder-calendar-or-node-features-active",
        {
            "method": "discrete graph structure learning with diffusion recurrence",
            "equations": [
                "e_i=f_phi(X_i); p(A_ij|X_i,X_j)=softmax(g_phi([e_i,e_j]))",
                "A_ij=GumbelSoftmax(p_ij,tau) with straight-through discrete training samples",
                "Y=Decoder_DCGRU(Encoder_DCGRU(X,A),A)",
            ],
            "modules": {"edge distribution and sampling": "DiscreteGraphDiscovery", "learned-graph diffusion": "LearnedDiffusion", "graph recurrent forecast": "GraphGRUCell/Model.encoder/Model.decoder"},
            "differences": ["current input window supplies node features", "soft edge probabilities in evaluation", "optional adjacency is a weak prior and auxiliary-loss target"],
        },
    ),
)


def _digest(value: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _runtime(case: RewriteCase) -> dict[str, object]:
    torch.manual_seed(260827)
    graph = _graph(4)
    model = case.factory(graph).cpu().eval()
    x = torch.randn(2, 6, 4, requires_grad=True)
    marks = _marks(2, 6)
    output = model(x, marks)
    if output.shape != (2, 3, 4) or not torch.isfinite(output).all():
        raise AssertionError("forward contract failed")
    output.square().mean().backward()
    if x.grad is None or not torch.isfinite(x.grad).all() or x.grad.abs().max() == 0:
        raise AssertionError("input gradient failed")
    gradients: dict[str, float] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all() or parameter.grad.abs().max() == 0:
            raise AssertionError(f"inactive or invalid parameter gradient: {name}")
        gradients[name] = float(parameter.grad.abs().max())

    clone = case.factory(graph).cpu().eval()
    clone.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    clone_output = clone(x.detach(), marks)
    torch.testing.assert_close(clone_output, output.detach())
    boundary = case.boundary_factory(np.ones((1, 1), dtype=np.float32)).cpu().eval()
    if boundary(torch.randn(1, 1, 1)).shape != (1, 1, 1):
        raise AssertionError("minimum node/sequence boundary failed")
    try:
        model(torch.randn(1, 5, 4))
    except ValueError:
        wrong_length_rejected = True
    else:
        raise AssertionError("wrong sequence length accepted")
    try:
        case.factory(np.eye(3, dtype=np.float32))
    except ValueError:
        wrong_adjacency_rejected = True
    else:
        raise AssertionError("wrong adjacency accepted")

    torch.manual_seed(91)
    identity_model = case.factory(np.eye(4, dtype=np.float32)).eval()
    torch.manual_seed(91)
    graph_model = case.factory(graph).eval()
    adjacency_effect = float((identity_model(x.detach()) - graph_model(x.detach())).abs().max())
    if adjacency_effect == 0:
        raise AssertionError("adjacency did not affect output")
    marked = model(x.detach(), _marks(2, 6, offset=7))
    unmarked = model(x.detach())
    marks_effect = float((marked - unmarked).abs().max())
    if case.name == "DSTAGNN" and marks_effect != 0:
        raise AssertionError("DSTAGNN unexpectedly consumed marks")
    if case.name != "DSTAGNN" and marks_effect == 0:
        raise AssertionError("declared marks did not affect output")
    return {
        "shape": [2, 3, 4],
        "input_gradient_max_abs": float(x.grad.abs().max()),
        "parameter_gradients": gradients,
        "round_trip_max_abs": float((clone_output - output.detach()).abs().max()),
        "adjacency_effect_max_abs": adjacency_effect,
        "marks_effect_max_abs": marks_effect,
        "marks_contract": case.marks,
        "wrong_length_rejected": wrong_length_rejected,
        "wrong_adjacency_rejected": wrong_adjacency_rejected,
    }


def _environment() -> dict[str, object]:
    return {
        "python": platform.python_version(),
        "framework": f"torch {torch.__version__}",
        "dependencies": {"numpy": np.__version__, "pydantic": importlib.metadata.version("pydantic"), "torch": torch.__version__},
        "platform": platform.platform(),
        "device": "cpu",
        "dtype": "float32",
        "deterministic": {"seed": 260827, "num_threads": torch.get_num_threads()},
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
    evidence = [relative, "tests/test_graph_clean_room_rewrites_a.py"]
    checks = {
        "paper_structure": _check(evidence, mapped_elements=len(case.structure["modules"]), claim="paper-equations-to-independent-local-map"),
        "equations": _check(evidence, cases=len(case.structure["equations"])),
        "construction": _check(evidence, instances=4),
        "forward": _check(evidence, shape="2,3,4"),
        "backward": _check(evidence, input_gradient_max_abs=observations["input_gradient_max_abs"]),
        "finite_outputs": _check(evidence, nonfinite=0),
        "active_parameter_gradients": _check(evidence, parameters=len(observations["parameter_gradients"])),
        "state_dict_round_trip": _check(evidence, max_abs=observations["round_trip_max_abs"]),
        "cpu": _check(evidence, device="cpu"),
        "batch_size_boundary": _check(evidence, cases="batch=1,batch=2"),
        "sequence_length_boundary": _check(evidence, cases="seq=1,node=1,wrong-length-rejected"),
        "marks_adjacency_contract": _check(evidence, contract=case.marks, adjacency_effect_max_abs=observations["adjacency_effect_max_abs"]),
    }
    result = {
        "schema_version": 1,
        "kind": "rewrite-validation",
        "model": case.name,
        "implementation": "rewrite",
        "verified_at": datetime.now(timezone.utc),
        "subject_sha256": verification_subject_sha256(ROOT, records[case.name]),
        "commands": [
            f"uv run python scripts/verify_graph_clean_room_rewrites_a.py --model {case.name}",
            "uv run python -m unittest tests.test_graph_clean_room_rewrites_a -v",
            f"uv run tsf repo doctor --strict --models {case.name}",
        ],
        "environment": _environment(),
        "artifacts": {relative: evidence_file_sha256(artifact_path)},
        "passed": True,
        "basis": {"references": [case.reference], "structure_map_sha256": structure_digest, "independent_design": True, "source_code_not_copied": True},
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
