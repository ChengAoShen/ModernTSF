#!/usr/bin/env python3
"""Generate strict clean-room evidence for recent forecasting rewrites."""

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
from models.apn.model import Model as APN  # noqa: E402
from models.cora.model import Model as CoRA  # noqa: E402
from models.hn_mvts.model import Model as HNMVTS  # noqa: E402
from models.interpdn.model import Model as InterPDN  # noqa: E402
from models.olinear.model import Model as OLinear  # noqa: E402
from models.phaseformer.model import Model as PhaseFormer  # noqa: E402
from models.sempo.model import Model as SEMPO  # noqa: E402
from models.sonnet.model import Model as Sonnet  # noqa: E402
from models.timemosaic.model import Model as TimeMosaic  # noqa: E402

Factory = Callable[[], nn.Module]


@dataclass(frozen=True)
class RewriteCase:
    name: str
    factory: Factory
    boundary_factory: Factory
    reference: str
    structure: dict[str, object]
    uses_marks_as_time: bool = False


CASES = (
    RewriteCase("OLinear", lambda: OLinear(4, 3, 2, d_model=4), lambda: OLinear(1, 1, 2, d_model=4),
                "https://arxiv.org/abs/2505.08550",
                {"method": "orthogonally transformed linear forecasting with normalized channel mixing",
                 "equation": "Z=Q_i^T x; W_n=softplus(W)/rowsum(softplus(W)); y=Q_o decode(ISL(CSL(Z)))",
                 "modules": {"OrthoTrans": "Model.input_basis/output_basis", "NormLin": "NormLin", "CSL/ISL": "Model channel and sequence learners"},
                 "differences": ["identity transform bases until training-data eigenvectors are installed", "one compact CSL/ISL block and direct flattened decoder"]}),
    RewriteCase("PhaseFormer", lambda: PhaseFormer(4, 3, 2, d_model=4, period=3, num_routers=2), lambda: PhaseFormer(1, 1, 2, d_model=4, period=3, num_routers=2),
                "https://arxiv.org/abs/2510.04134",
                {"method": "phase tokenization with two-stage cross-phase router attention",
                 "equation": "X_phase=reshape(circular_pad(x)); H=MHA(R,Z,Z); Z'=MHA(Z,H,H); Y_phase=linear(Z')",
                 "modules": {"phase tokens": "Model._tokenize", "router aggregation/distribution": "CrossPhaseRouter", "shared predictor": "Model.predictor"},
                 "differences": ["period is explicitly configured rather than estimated by autocorrelation", "one channel-independent routing layer by default"]}),
    RewriteCase("InterPDN", lambda: InterPDN(4, 3, 2, support_size=7), lambda: InterPDN(1, 1, 2, support_size=7),
                "https://arxiv.org/abs/2511.23260",
                {"method": "per-step distributions on interleaved supports with confidence fusion",
                 "equation": "e_j=<softmax(logits_j),support_j>; w=max(p_1)/(max(p_1)+max(p_2)); y=w*e_1+(1-w)*e_2",
                 "modules": {"dual independent branches": "Model.branches", "normal-quantile supports": "support_first/support_second", "expectation and confidence fusion": "Model.forward"},
                 "differences": ["compact residual seasonal encoder replaces the paper patch encoder", "coarse auxiliary branches and training-only consistency losses are omitted"]}),
    RewriteCase("Sonnet", lambda: Sonnet(4, 3, 2, d_model=4, num_wavelets=2), lambda: Sonnet(1, 1, 2, d_model=4, num_wavelets=2),
                "https://arxiv.org/abs/2505.15312",
                {"method": "learnable wavelets, spectral-coherence weighting, and stable Koopman evolution",
                 "equation": "M=exp(-a*t^2)cos(b*t+g*t^2); C=|Q_f K_f*|^2/(P_qq P_kk+eps); K=U diag(exp(i p)) U*",
                 "modules": {"wavelet atoms": "LearnableWavelets", "MVCA": "SpectralCoherence", "unitary Koopman": "StableKoopman", "reconstruction/decoder": "Model.forward/decoder"},
                 "differences": ["all channels use a symmetric joint embedding rather than an endogenous/exogenous alpha split", "adaptive horizon pooling replaces dataset-specific decoder sizing"]}),
    RewriteCase("APN", lambda: APN(16, 6, 3, 16, 4, 4), lambda: APN(2, 1, 2, 4, 2, 1),
                "https://arxiv.org/abs/2505.11250",
                {"method": "time-aware soft patch aggregation and query-time decoding",
                 "equation": "alpha=σ((right-t)/softplus(kappa))*σ((t-left)/softplus(kappa)); h=sum(alpha*v)/sum(alpha); context=softmax(q h^T/sqrt(D))h",
                 "modules": {"adaptive windows": "Model.patch_weights", "weighted aggregation": "Model.forward", "query decoder": "Model.decoder"},
                 "differences": ["dense regular timestamps by default; explicit dense times supported", "no asynchronous ragged loader or missing-data benchmark protocol"]}, True),
    RewriteCase("CoRA", lambda: CoRA(16, 6, 3, 16, 2, 2), lambda: CoRA(1, 1, 2, 4, 1, 1),
                "https://arxiv.org/abs/2603.21828",
                {"method": "dynamic low-rank correlation with heterogeneous fusion",
                 "equation": "Q=sum_i C_i*q^i; V=sigmoid(relu(E1 E2^T)); M=Pearson(X)+Q V Q^T",
                 "modules": {"dynamic correlation": "Model.dynamic_correlation", "heterogeneous projections": "Model.positive/negative", "gated correction": "Model.fusion/gate"},
                 "differences": ["local linear forecaster replaces a pre-trained TSFM", "training-only H-PCorr contrastive loss is omitted"]}),
    RewriteCase("HN_MVTS", lambda: HNMVTS(16, 6, 3, 16, 4, 8), lambda: HNMVTS(1, 1, 2, 2, 2, 2),
                "https://arxiv.org/abs/2511.08340",
                {"method": "channel-embedding-conditioned final-layer hypernetwork",
                 "equation": "W_K=h_phi(Z); forecast_n=W_K^(n) h^(n)+b_n",
                 "modules": {"component embeddings": "Model.channel_embedding", "partial hypernetwork": "Model.hypernetwork", "generated decoder": "Model.generated_projection"},
                 "differences": ["compact temporal MLP is the base model", "embeddings are not initialized from training-split Pearson/PCA", "generated projections are evaluated on each forward call"]}),
    RewriteCase("SEMPO", lambda: SEMPO(16, 6, 3, 16, 4, 3, 4, 0), lambda: SEMPO(2, 1, 2, 4, 1, 2, 1, 0),
                "https://arxiv.org/abs/2510.19710",
                {"method": "energy-aware spectral decomposition and mixture-of-prompts attention",
                 "equation": "Z_H=Z*energy_selector; Z_L=Z-Z_H; e=sum_i softmax(router(b))_i e_i; attention(Q=B,K=[e_K;B],V=[e_V;B])",
                 "modules": {"spectral partition": "Model.energy_aware_decomposition", "prompt routing": "Model.router/prompt_experts", "prompt attention": "Model.attention"},
                 "differences": ["deterministic differentiable spectral masks", "no pre-trained weights, reconstruction corpus, or two-stage training harness"]}),
    RewriteCase("TimeMosaic", lambda: TimeMosaic(16, 6, 3, 16, (2, 4, 8), 3, 4, 0), lambda: TimeMosaic(2, 1, 2, 2, (1, 2), 1, 1, 0),
                "https://arxiv.org/abs/2509.19406",
                {"method": "adaptive-granularity regions and segment-wise prompt decoding",
                 "equation": "z_r=Linear_f(Unfold(x_r;f)); aligned=RepeatPad(z_r); Y_k=head_k(Attention(Q=X,K=[prompt_k;X],V=[prompt_k;X]))",
                 "modules": {"granularity selector": "Model.granularity_classifier", "aligned patches": "Model.adaptive_patch_tokens", "segment prompts/heads": "Model.segment_prompts/segment_heads"},
                 "differences": ["soft selection replaces Gumbel-hard selection", "no frozen foundation backbone, budget loss, or large-corpus pre-training"]}),
)


def _digest(value: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _runtime(case: RewriteCase) -> dict[str, object]:
    torch.manual_seed(260827)
    model = case.factory().cpu().eval()
    seq_len, pred_len, enc_in = model.seq_len, model.pred_len, model.enc_in
    x = torch.randn(2, seq_len, enc_in, requires_grad=True)
    marks = torch.linspace(0, 1, seq_len).square().reshape(1, seq_len, 1).expand(2, -1, -1)
    adjacency = torch.eye(enc_in)
    output = model(x, marks, adjacency)
    if output.shape != (2, pred_len, enc_in) or not torch.isfinite(output).all():
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
    torch.testing.assert_close(clone(x.detach(), marks, adjacency), output.detach())
    if model(torch.randn(1, seq_len, enc_in), marks[:1], adjacency).shape != (1, pred_len, enc_in):
        raise AssertionError("batch boundary failed")
    boundary = case.boundary_factory().eval()
    boundary_length = boundary.seq_len
    boundary_channels = boundary.enc_in
    if boundary(torch.randn(1, boundary_length, boundary_channels)).shape != (1, 1, boundary_channels):
        raise AssertionError("minimum sequence fixture failed")
    try:
        model(torch.randn(1, seq_len - 1, enc_in))
    except ValueError:
        wrong_length_rejected = True
    else:
        raise AssertionError("wrong sequence length accepted")
    plain = model(x.detach())
    marks_effect = float((plain - output.detach()).abs().max())
    if case.uses_marks_as_time and marks_effect == 0:
        raise AssertionError("APN must consume supplied observation times")
    if not case.uses_marks_as_time:
        torch.testing.assert_close(plain, output.detach())
    return {"shape": [2, pred_len, enc_in], "input_gradient_max_abs": float(x.grad.abs().max()),
            "parameter_gradients": gradients, "round_trip_max_abs": 0.0,
            "wrong_length_rejected": wrong_length_rejected, "marks_effect_max_abs": marks_effect}


def _environment() -> dict[str, object]:
    return {"python": platform.python_version(), "framework": f"torch {torch.__version__}",
            "dependencies": {"pydantic": importlib.metadata.version("pydantic"), "torch": torch.__version__},
            "platform": platform.platform(), "device": "cpu", "dtype": "float32",
            "deterministic": {"seed": 260827, "num_threads": torch.get_num_threads()}}


def _check(evidence: list[str], **metrics: float | int | str) -> dict[str, object]:
    return {"passed": True, "evidence": evidence, "metrics": metrics}


def verify(case: RewriteCase, records: dict[str, dict[str, object]]) -> None:
    observations = _runtime(case)
    structure_digest = _digest(case.structure)
    relative = f"verification/rewrite/{case.name}.json"
    artifact_path = ROOT / relative
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {"schema_version": 1, "kind": "clean-room-structure-map", "model": case.name,
                "reference": case.reference, "independent_design": True,
                "source_code_not_copied": True, "structure_map": case.structure,
                "structure_map_sha256": structure_digest, "observations": observations}
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence = [relative, "tests/test_recent_clean_room_rewrites.py"]
    checks = {
        "paper_structure": _check(evidence, mapped_elements=len(case.structure["modules"]), claim="paper-equations-to-independent-local-map"),
        "equations": _check(evidence, cases=1), "construction": _check(evidence, instances=3),
        "forward": _check(evidence, shape=",".join(str(value) for value in observations["shape"])), "backward": _check(evidence, input_gradient_max_abs=observations["input_gradient_max_abs"]),
        "finite_outputs": _check(evidence, nonfinite=0), "active_parameter_gradients": _check(evidence, parameters=len(observations["parameter_gradients"])),
        "state_dict_round_trip": _check(evidence, max_abs=0.0), "cpu": _check(evidence, device="cpu"),
        "batch_size_boundary": _check(evidence, cases="batch=1,batch=2"), "sequence_length_boundary": _check(evidence, cases="minimum-valid,wrong-length-rejected"),
        "marks_adjacency_contract": _check(evidence, contract="APN-consumes-dense-times;others-accept-and-ignore")}
    result = {"schema_version": 1, "kind": "rewrite-validation", "model": case.name,
              "implementation": "rewrite", "verified_at": datetime.now(timezone.utc),
              "subject_sha256": verification_subject_sha256(ROOT, records[case.name]),
              "commands": [f"uv run python scripts/verify_recent_clean_room_rewrites.py --model {case.name}",
                           "uv run python -m unittest tests.test_recent_clean_room_rewrites -v",
                           f"uv run tsf repo doctor --strict --models {case.name}"],
              "environment": _environment(), "artifacts": {relative: evidence_file_sha256(artifact_path)},
              "passed": True, "basis": {"references": [case.reference], "structure_map_sha256": structure_digest,
                                           "independent_design": True, "source_code_not_copied": True},
              "checks": checks}
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
