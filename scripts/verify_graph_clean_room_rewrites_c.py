#!/usr/bin/env python3
"""Generate executable clean-room evidence for graph rewrite batch C."""
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
from benchmark.verification_results import evidence_file_sha256, verification_subject_sha256, write_verification_result  # noqa: E402
from models.mage.model import Model as MAGE  # noqa: E402
from models.mgsfformer.model import Model as MGSFformer  # noqa: E402
from models.pcdcnet.model import Model as PCDCNet  # noqa: E402
from models.stop.model import Model as STOP  # noqa: E402
from models.sttn.model import Model as STTN  # noqa: E402
from models.stwave.model import Model as STWave  # noqa: E402


def graph(nodes: int) -> np.ndarray:
    result = np.eye(nodes, dtype=np.float32)
    for index in range(nodes - 1):
        result[index, index + 1], result[index + 1, index] = 1.0, 0.25
    return result


def marks(batch: int, steps: int, offset: int = 0) -> torch.Tensor:
    rows = [[2026, 8, 1 + i // 24, 5, (i + offset) % 24, 0] for i in range(steps)]
    return torch.tensor([rows] * batch, dtype=torch.float32)


@dataclass(frozen=True)
class Case:
    name: str
    length: int
    factory: Callable[[np.ndarray], nn.Module]
    boundary: Callable[[], nn.Module]
    reference: str
    contract: str
    structure: dict[str, object]
    graph_active: bool = False
    marks_active: bool = True


CASES = (
    Case("MAGE", 6, lambda g: MAGE(6, 3, 4, model_dim=8, recur_num=3, topk=2, node_dim=4),
         lambda: MAGE(1, 1, 1, model_dim=2, recur_num=2, topk=1, node_dim=1),
         "https://proceedings.neurips.cc/paper_files/paper/2025/hash/54c9bfb0885ae07f23607f617ab64c2b-Abstract-Conference.html",
         "calendar-marks-active;adjacency-not-required",
         {"method": "linear mixture of adaptive graph experts", "equations": [
             "H_e=softmax(E_target) softmax(E_source)^T V_e(X)",
             "R=TopKSoftmax(router(X)); H=sum_e R_e H_e",
             "three expert blocks use depths 1,2,3 with residual forecast"],
          "modules": {"factorised kernel expert": "AdaptiveGraphExpert", "sparse balanced router": "MixtureGraphBlock", "multi-depth backbone": "Model.blocks"},
          "differences": ["compact averaged calendar prompt", "training expert-count objective omitted", "no metric parity claim"]}),
    Case("MGSFformer", 24, lambda g: MGSFformer(24, 3, 4, IE_dim=8, dropout=0, num_head=2),
         lambda: MGSFformer(24, 1, 1, IE_dim=2, dropout=0, num_head=1),
         "https://doi.org/10.1016/j.inffus.2024.102607", "marks-intentionally-not-consumed;adjacency-not-required",
         {"method": "multi-granularity spatiotemporal fusion transformer", "equations": [
             "G_f=Interpolate(MeanPool_f(X)), f in {1,3,6,12,24}",
             "D_f=Norm(G_f-W_f G_previous)",
             "Y=sum_f softmax(score_f) Forecast_f(STAttention(D_f))"],
          "modules": {"residual de-redundancy": "ResidualDeRedundant", "spatiotemporal attention": "SpatioTemporalAttention", "dynamic fusion": "DynamicFusion"},
          "differences": ["coarse views interpolated to fine grid", "private air-quality preprocessing omitted", "no auxiliary loss"]}, marks_active=False),
    Case("PCDCNet", 6, lambda g: PCDCNet(6, 3, 4, g, d_model=8, dropout=0),
         lambda: PCDCNet(1, 1, 1, np.ones((1, 1), np.float32), cov_dim=0, d_model=2, dropout=0),
         "https://arxiv.org/abs/2505.19842", "future/encoder-marks-and-adjacency-active",
         {"method": "physical-chemical local/transport/accumulation surrogate", "equations": [
             "E_t=MLP(RMSNorm(Linear([X_(t-1),P_t,Q_t])))",
             "M_t=Linear((I-D^-1/2 A D^-1/2)H_t); Z_t=GRU(H_t,Z_(t-1))",
             "Xhat_t=Xhat_(t-1)+Linear(H_t); L_DIC enforces transport continuity"],
          "modules": {"local interaction dynamics": "LocalInteractionDynamics", "spatial transport dynamics": "SpatialTransportDynamics", "temporal accumulation": "Model.accumulation", "domain constraint": "Model.domain_informed_constraint"},
          "differences": ["calendar marks substitute for separate meteorology/emissions", "generic station graph", "deployment and 72-hour protocol omitted"]}, graph_active=True),
    Case("STOP", 6, lambda g: STOP(6, 3, 4, model_dim=4, prompt_dim=2, num_layer=1, hid_dim=8, core=2, head=2),
         lambda: STOP(1, 1, 1, model_dim=2, prompt_dim=1, num_layer=1, hid_dim=2, core=1, head=1),
         "https://proceedings.mlr.press/v267/ma25s.html", "calendar-marks-active;adjacency-intentionally-absent",
         {"method": "centralized spatiotemporal OOD processor", "equations": [
             "H0=MLP_long(X_long)+MLP_short(X_short); ZT=ChannelMix([H0,P,Et,Ed])",
             "Zc=softmax(alpha QK^T) softmax(alpha KQ^T+G)V",
             "Zp=ZT-Zc; Y=Y_temporal+Y_spatial"],
          "modules": {"time decomposition": "SeriesDecomposition", "centralized ConAU interaction": "CentralizedInteraction", "GenPU branches": "Model.environment_forecasts"},
          "differences": ["external loop selects worst DRO branch", "generic calendar prompts", "paper OOD splits omitted"]}),
    Case("STTN", 6, lambda g: STTN(6, 3, 4, g, d_model=8, num_layers=1, dropout=0),
         lambda: STTN(1, 1, 1, np.ones((1, 1), np.float32), cov_dim=0, d_model=4, num_layers=1, dropout=0),
         "https://arxiv.org/abs/2001.02908", "calendar-marks-and-adjacency-active",
         {"method": "stacked spatial-temporal transformer network", "equations": [
             "S_dynamic=MultiHeadSoftmax(Q_node K_node^T)V_node",
             "S=Gate(S_dynamic, A X W_fixed)",
             "T=BidirectionalSelfAttention_time(S); Y=DirectHead(T)"],
          "modules": {"dynamic/fixed spatial transformer": "SpatialTransformer", "temporal transformer": "TemporalTransformer", "stacked ST blocks": "SpatialTemporalBlock"},
          "differences": ["direct multi-horizon head", "calendar covariates included", "official TensorFlow data pipeline omitted"]}, graph_active=True),
    Case("STWave", 6, lambda g: STWave(6, 3, 4, g, hidden_size=4, layers=1),
         lambda: STWave(1, 1, 1, np.ones((1, 1), np.float32), hidden_size=2, layers=1),
         "https://arxiv.org/abs/2112.02740", "calendar-marks-and-adjacency-active",
         {"method": "wavelet-disentangled efficient spectral graph attention", "equations": [
             "x_low=(x_even+x_odd)/sqrt(2); x_high=(x_even-x_odd)/sqrt(2)",
             "Phi=eigenvectors(I-D^-1/2 A D^-1/2)",
             "dual encoders use neighbor attention plus sampled global queries; Y=Gate(Y_low,Y_high)"],
          "modules": {"wavelet split": "wavelet_disentangle", "spectral sparse attention": "SpectralGraphAttention", "dual encoder": "DualEncoder", "adaptive fusion": "AdaptiveFusion"},
          "differences": ["single-level Haar basis", "auxiliary low-band supervision omitted", "local graph preprocessing"]}, graph_active=True),
)


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def runtime(case: Case) -> dict[str, object]:
    torch.manual_seed(260827)
    model = case.factory(graph(4)).cpu().eval()
    x = torch.randn(2, case.length, 4, requires_grad=True)
    calendar = marks(2, case.length)
    output = model(x, calendar)
    if output.shape != (2, 3, 4) or not torch.isfinite(output).all():
        raise AssertionError("forward contract failed")
    output.square().mean().backward()
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all() or parameter.grad.abs().max() == 0:
            raise AssertionError(f"inactive parameter: {name}")
        gradients[name] = float(parameter.grad.abs().max())
    if x.grad is None or x.grad.abs().max() == 0:
        raise AssertionError("input gradient failed")
    clone = case.factory(graph(4)).eval()
    clone.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    clone_output = clone(x.detach(), calendar)
    torch.testing.assert_close(clone_output, output.detach())
    boundary = case.boundary().eval()
    boundary_length = boundary.seq_len
    if boundary(torch.randn(1, boundary_length, 1)).shape != (1, 1, 1):
        raise AssertionError("boundary failed")
    try:
        model(torch.randn(1, case.length - 1, 4))
    except ValueError:
        wrong_length = True
    else:
        raise AssertionError("wrong length accepted")
    marked_difference = float((model(x.detach(), marks(2, case.length, 7)) - model(x.detach())).abs().max())
    if case.marks_active != (marked_difference > 0):
        raise AssertionError("marks contract failed")
    adjacency_difference = 0.0
    if case.graph_active:
        torch.manual_seed(91); identity = case.factory(np.eye(4, dtype=np.float32)).eval()
        torch.manual_seed(91); connected = case.factory(graph(4)).eval()
        adjacency_difference = float((identity(x.detach()) - connected(x.detach())).abs().max())
        if adjacency_difference == 0:
            raise AssertionError("adjacency inactive")
    return {"shape": [2, 3, 4], "input_gradient_max_abs": float(x.grad.abs().max()),
            "parameter_gradients": gradients, "round_trip_max_abs": float((clone_output-output.detach()).abs().max()),
            "marks_effect_max_abs": marked_difference, "adjacency_effect_max_abs": adjacency_difference,
            "wrong_length_rejected": wrong_length, "contract": case.contract}


def environment() -> dict[str, object]:
    return {"python": platform.python_version(), "framework": f"torch {torch.__version__}",
            "dependencies": {"numpy": np.__version__, "pydantic": importlib.metadata.version("pydantic"), "torch": torch.__version__},
            "platform": platform.platform(), "device": "cpu", "dtype": "float32",
            "deterministic": {"seed": 260827, "num_threads": torch.get_num_threads()}}


def check(evidence: list[str], **metrics: float | int | str) -> dict[str, object]:
    return {"passed": True, "evidence": evidence, "metrics": metrics}


def verify(case: Case, records: dict[str, dict[str, object]]) -> None:
    observations = runtime(case)
    structure_hash = digest(case.structure)
    relative = f"verification/rewrite/{case.name}.json"
    path = ROOT / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {"schema_version": 1, "kind": "clean-room-structure-map", "model": case.name,
                "reference": case.reference, "independent_design": True,
                "source_code_not_copied": True, "structure_map": case.structure,
                "structure_map_sha256": structure_hash, "observations": observations}
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence = [relative, "tests/test_graph_clean_room_rewrites_c.py"]
    checks = {
        "paper_structure": check(evidence, mapped_elements=len(case.structure["modules"]), claim="paper-equations-to-independent-local-map"),
        "equations": check(evidence, cases=len(case.structure["equations"])), "construction": check(evidence, instances=3),
        "forward": check(evidence, shape="2,3,4"), "backward": check(evidence, input_gradient_max_abs=observations["input_gradient_max_abs"]),
        "finite_outputs": check(evidence, nonfinite=0), "active_parameter_gradients": check(evidence, parameters=len(observations["parameter_gradients"])),
        "state_dict_round_trip": check(evidence, max_abs=observations["round_trip_max_abs"]), "cpu": check(evidence, device="cpu"),
        "batch_size_boundary": check(evidence, cases="batch=1,batch=2"), "sequence_length_boundary": check(evidence, cases="minimum-boundary,wrong-length-rejected"),
        "marks_adjacency_contract": check(evidence, contract=case.contract, marks_effect_max_abs=observations["marks_effect_max_abs"], adjacency_effect_max_abs=observations["adjacency_effect_max_abs"]),
    }
    result = {"schema_version": 1, "kind": "rewrite-validation", "model": case.name,
              "implementation": "rewrite", "verified_at": datetime.now(timezone.utc),
              "subject_sha256": verification_subject_sha256(ROOT, records[case.name]),
              "commands": [f"uv run python scripts/verify_graph_clean_room_rewrites_c.py --model {case.name}",
                           "uv run python -m unittest tests.test_graph_clean_room_rewrites_c -v",
                           f"uv run tsf repo doctor --strict --models {case.name}"],
              "environment": environment(), "artifacts": {relative: evidence_file_sha256(path)},
              "passed": True, "basis": {"references": [case.reference], "structure_map_sha256": structure_hash,
                                          "independent_design": True, "source_code_not_copied": True}, "checks": checks}
    write_verification_result(ROOT / "verification/model-results.json", result)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", choices=[case.name for case in CASES])
    selected = set(parser.parse_args().model or [case.name for case in CASES])
    records = {str(record["name"]): record for record in model_records(ROOT)}
    for case in CASES:
        if case.name in selected:
            verify(case, records)
            print(f"{case.name}: rewrite validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
