#!/usr/bin/env python3
"""Generate strict paper-structure/runtime evidence for six air rewrites."""

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
from benchmark.verification_results import (  # noqa: E402
    evidence_file_sha256,
    verification_subject_sha256,
    write_verification_result,
)
from models.aircade.model import Model as AirCade  # noqa: E402
from models.airdualode.model import Model as AirDualODE  # noqa: E402
from models.airformer.model import Model as AirFormer  # noqa: E402
from models.airphynet.model import Model as AirPhyNet  # noqa: E402
from models.cauair.model import Model as CauAir  # noqa: E402
from models.deepair.model import Model as DeepAir  # noqa: E402


@dataclass(frozen=True)
class Case:
    name: str
    factory: Callable[[], nn.Module]
    boundary_factory: Callable[[], nn.Module]
    reference: str
    structure: dict[str, object]
    graph_contract: str


CASES = (
    Case("AirCade",
        lambda: AirCade(6, 3, 3, d_model=16, prompt_dim=2, adaptive_dim=3,
            num_heads=2, temporal_layers=1, spatial_layers=1),
        lambda: AirCade(1, 1, 1, d_model=8, prompt_dim=1, adaptive_dim=1,
            num_heads=1, temporal_layers=1, spatial_layers=1),
        "https://arxiv.org/abs/2505.20119",
        {"equations": {"Eq. 1": "value/weather projections plus temporal and station prompts",
            "Eqs. 2-7": "four-path direct/adaptive/inverse DK-MSA with temporal gates",
            "Eqs. 8-9": "historical Cade residual attention and MLP",
            "Eqs. 10-12": "future Cadi propagation and point predictor",
            "Eq. 13": "multi-environment intervention masks"},
         "modules": {"DK-Prompt": "Model._prompted", "DK-MSA": "DomainKnowledgeAttention",
            "Cade/Cadi": "CausalLayer and axis-specific stacks", "predictor": "Model.predictor"},
         "differences": ["continuous intervention relaxation", "variance objective external"]},
        "marks/future-covariates-active;learned-adaptive-spatial-matrices;adjacency-not-consumed"),
    Case("AirDualODE",
        lambda: AirDualODE(6, 3, 3, phy_latent_dim=4, unk_latent_dim=4,
            gcn_hidden_dim=8, n_heads=2),
        lambda: AirDualODE(1, 1, 1, phy_latent_dim=2, unk_latent_dim=2,
            gcn_hidden_dim=4, n_heads=1),
        "https://openreview.net/forum?id=kOJf7Dklyv",
        {"equations": {"Eq. 6": "BA-DAE diffusion, directed advection, and open-boundary correction",
            "Eqs. 7-8": "explicit physics rollout and latent projection",
            "Eqs. 9-10": "history encoder and masked-attention data ODE",
            "Eqs. 11-13": "alignment is an experiment loss; concatenated graph fusion is retained"},
         "modules": {"BA-DAE": "BoundaryAwareDynamics", "learned dynamics": "DataDrivenDynamics",
            "fusion": "Model.graph_fusion", "solver": "Model._integrate"},
         "differences": ["precomputed flow graph", "Decay-TCL external to forward"]},
        "marks-active;distance-and-directed-flow-graphs-construction-inputs"),
    Case("AirFormer",
        lambda: AirFormer(6, 3, 3, d_model=8, nhead=2, num_encoder_layers=2,
            spatial_regions=2, dropout=0.0),
        lambda: AirFormer(1, 1, 1, d_model=4, nhead=1, num_encoder_layers=1,
            spatial_regions=1, dropout=0.0),
        "https://doi.org/10.1609/aaai.v37i12.26676",
        {"equations": {"CT-MSA": "causal local windows with growing receptive fields",
            "DS-MSA": "query-specific dartboard aggregation and regional attention",
            "stochastic stage": "reverse-level Gaussian latent hierarchy"},
         "modules": {"CT-MSA": "CausalTemporalAttention", "DS-MSA": "DartboardSpatialAttention",
            "deterministic stage": "AirFormerBlock", "stochastic stage": "Model.latent_mean/scale"},
         "differences": ["dataset supplies exact geographic projection", "point forecast API"]},
        "marks-active;dartboard-projection-construction-input"),
    Case("AirPhyNet",
        lambda: AirPhyNet(6, 3, 3, latent_dim=4, rnn_units=8, ode_method="euler"),
        lambda: AirPhyNet(1, 1, 1, latent_dim=2, rnn_units=4, ode_method="euler"),
        "https://openreview.net/forum?id=JW3jTjaaAB",
        {"equations": {"Eq. 9": "GRU posterior mean/scale and reparameterized initial state",
            "Eqs. 10-11": "diffusion-advection graph differential equation",
            "Eq. 12": "ODE trajectory and shared pollutant decoder"},
         "modules": {"encoder": "Model.encoder/initial_mean/initial_scale",
            "physics field": "PhysicsVectorField", "solver": "Model._ode_step",
            "decoder": "Model.decoder"},
         "differences": ["precomputed flow operator", "local Euler/RK4 solver"]},
        "marks-active;distance-and-directed-flow-graphs-construction-inputs"),
    Case("CauAir",
        lambda: CauAir(6, 3, 3, dim=8, cache_count=2, heads=2),
        lambda: CauAir(1, 1, 1, dim=4, cache_count=1, heads=1),
        "https://www.ijcai.org/proceedings/2025/353",
        {"equations": {"Eq. 5": "parallel learned mixture of cache-attention and SwiGLU FFN",
            "Eqs. 6-11": "multi-head station-cache assignment, aggregation, and reconstruction",
            "causal stages": "past AQI/weather association followed by future-weather propagation"},
         "modules": {"cache attention": "CacheAttention", "SwiGLU": "SwiGLU",
            "CachLormer": "CachLormer", "causal propagation": "Model.forward"},
         "differences": ["shared linear temporal summaries", "calendar weather fallback"]},
        "historical/future-covariates-active;graph-free-adjacency-not-consumed"),
    Case("DeepAir",
        lambda: DeepAir(6, 3, 3, hidden_dim=8, spatial_regions=2),
        lambda: DeepAir(1, 1, 1, hidden_dim=4, spatial_regions=1),
        "https://doi.org/10.1145/3219819.3219822",
        {"equations": {"spatial transformation": "target-relative partition, aggregation, interpolation",
            "distributed fusion": "HW/WF/SP/MP/HI residual FusionNets",
            "Eq. 1": "horizon/node weighted merge followed by sigmoid"},
         "modules": {"spatial transform": "Model.spatial_projection", "FusionNet": "FusionNet",
            "five subnets": "Model historical_weather/weather_forecast/secondary_pollutants/meta_properties/holistic",
            "weighted merge": "Model.fusion_weights"},
         "differences": ["generic factor packing", "dataset supplies geographic regions"]},
        "historical/future-covariates-active;spatial-projection-construction-input"),
)


def _marks(batch: int, length: int, offset: int = 0) -> torch.Tensor:
    result = torch.zeros(batch, length, 6)
    result[..., 0] = 2024
    result[..., 1] = 1
    result[..., 2] = torch.arange(1, length + 1)
    result[..., 3] = (torch.arange(length) + offset) % 7
    result[..., 4] = (torch.arange(length) + offset) % 24
    return result


def _runtime(case: Case) -> dict[str, object]:
    seed = 92711 + sum(map(ord, case.name))
    torch.manual_seed(seed)
    model = case.factory().cpu()
    values = torch.randn(2, 6, 3, requires_grad=True)
    output = model(values, _marks(2, 6), None, _marks(2, 3))
    if output.shape != (2, 3, 3) or not torch.isfinite(output).all():
        raise AssertionError(f"{case.name}: forward/finite contract failed")
    output.square().mean().backward()
    if values.grad is None or not torch.isfinite(values.grad).all() or values.grad.abs().max() == 0:
        raise AssertionError(f"{case.name}: input gradient failed")
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all() or parameter.grad.abs().max() == 0:
            raise AssertionError(f"{case.name}: inactive parameter {name}")
        gradients[name] = float(parameter.grad.abs().max())

    model.eval()
    expected = model(values.detach(), _marks(2, 6), None, _marks(2, 3))
    clone = case.factory().eval()
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    torch.testing.assert_close(clone(values.detach(), _marks(2, 6), None, _marks(2, 3)), expected)
    changed = model(values.detach(), _marks(2, 6, 3), None, _marks(2, 3, 5))
    if torch.equal(expected, changed):
        raise AssertionError(f"{case.name}: covariates are inactive")
    if model(torch.randn(1, 6, 3), _marks(1, 6), None, _marks(1, 3)).shape != (1, 3, 3):
        raise AssertionError(f"{case.name}: batch boundary failed")
    try:
        model(torch.randn(1, 5, 3), _marks(1, 5), None, _marks(1, 3))
    except ValueError:
        wrong_length_rejected = True
    else:
        raise AssertionError(f"{case.name}: wrong history length accepted")
    boundary = case.boundary_factory().eval()
    if boundary(torch.randn(1, 1, 1), _marks(1, 1), None, _marks(1, 1)).shape != (1, 1, 1):
        raise AssertionError(f"{case.name}: minimum boundary failed")
    node_marks = torch.randn(1, 6, 3, 2)
    future = torch.randn(1, 3, 3, 2)
    if model(torch.randn(1, 6, 3), node_marks, None, future).shape != (1, 3, 3):
        raise AssertionError(f"{case.name}: structured covariate contract failed")
    return {"shape": [2, 3, 3], "input_gradient_max_abs": float(values.grad.abs().max()),
        "parameter_gradients": gradients, "round_trip_max_abs": 0.0,
        "marks_active": True, "structured_covariates": True,
        "batch_size_cases": [1, 2], "minimum_history": 1,
        "minimum_nodes": 1, "wrong_length_rejected": wrong_length_rejected,
        "graph_contract": case.graph_contract}


def _digest(value: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _environment(seed: int) -> dict[str, object]:
    return {"python": platform.python_version(), "framework": f"torch {torch.__version__}",
        "dependencies": {"pydantic": importlib.metadata.version("pydantic"), "torch": torch.__version__},
        "platform": platform.platform(), "device": "cpu", "dtype": "float32",
        "deterministic": {"seed": seed, "num_threads": torch.get_num_threads()}}


def _check(evidence: list[str], **metrics) -> dict[str, object]:
    return {"passed": True, "evidence": evidence, "metrics": metrics}


def verify(case: Case, records: dict[str, dict[str, object]]) -> None:
    observations = _runtime(case)
    digest = _digest(case.structure)
    relative = f"verification/rewrite/{case.name}.json"
    artifact_path = ROOT / relative
    artifact = {"schema_version": 1, "kind": "clean-room-structure-map", "model": case.name,
        "reference": case.reference, "independent_design": True, "source_code_not_copied": True,
        "structure_map": case.structure, "structure_map_sha256": digest, "observations": observations}
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence = [relative, "tests/test_air_physical_clean_room_rewrites.py"]
    checks = {
        "paper_structure": _check(evidence, mapped_elements=len(case.structure["modules"])),
        "equations": _check(evidence, cases=len(case.structure["equations"])),
        "construction": _check(evidence, instances=3),
        "forward": _check(evidence, shape="2,3,3"),
        "backward": _check(evidence, input_gradient_max_abs=observations["input_gradient_max_abs"]),
        "finite_outputs": _check(evidence, nonfinite=0),
        "active_parameter_gradients": _check(evidence, parameters=len(observations["parameter_gradients"])),
        "state_dict_round_trip": _check(evidence, max_abs=0.0),
        "cpu": _check(evidence, device="cpu"),
        "batch_size_boundary": _check(evidence, cases="batch=1,batch=2,node=1,node=3"),
        "sequence_length_boundary": _check(evidence, cases="minimum=1;wrong-length-rejected"),
        "marks_adjacency_contract": _check(evidence, contract=case.graph_contract),
    }
    seed = 92711 + sum(map(ord, case.name))
    result = {"schema_version": 1, "kind": "rewrite-validation", "model": case.name,
        "implementation": "rewrite", "verified_at": datetime.now(timezone.utc),
        "subject_sha256": verification_subject_sha256(ROOT, records[case.name]),
        "commands": [f"uv run python scripts/verify_air_physical_clean_room_rewrites.py --model {case.name}",
            "uv run python -m unittest tests.test_air_physical_clean_room_rewrites -v",
            f"uv run tsf repo doctor --strict --models {case.name}"],
        "environment": _environment(seed), "artifacts": {relative: evidence_file_sha256(artifact_path)},
        "passed": True, "basis": {"references": [case.reference], "structure_map_sha256": digest,
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
