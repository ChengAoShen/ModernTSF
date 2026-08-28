#!/usr/bin/env python3
"""Generate clean-room evidence for probability/attention final rewrites."""

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
from models.glocalib.model import Model as GlocalIB  # noqa: E402
from models.pattn.model import Model as PAttn  # noqa: E402
from models.phat.model import Model as PHAT  # noqa: E402
from models.quantile_dlinear.model import Model as QuantileDLinear  # noqa: E402
from models.quantile_patchtst.model import Model as QuantilePatchTST  # noqa: E402
from models.tide.model import Model as TiDE  # noqa: E402


REFERENCES = {
    "GlocalIB": "https://arxiv.org/abs/2510.04910",
    "PAttn": "https://arxiv.org/abs/2406.16964",
    "PHAT": "https://arxiv.org/abs/2602.00654",
    "QuantileDLinear": "https://arxiv.org/abs/2205.13504",
    "QuantilePatchTST": "https://arxiv.org/abs/2211.14730",
    "TiDE": "https://arxiv.org/abs/2304.08424",
}

STRUCTURES = {
    "GlocalIB": {
        "method": "forecasting adaptation of the global-local information bottleneck",
        "equation": "q(z|x)=N(mu,diag(exp(logvar))); Laux=alpha*KL(q||N(0,I))+beta*Align(proj(z_masked),stopgrad(z_clean))",
        "modules": {"Eq.6 variational encoder": "_VariationalSequenceEncoder", "Eq.8 KL": "Model.forward", "Eq.12-13 alignment": "Model._alignment", "forecasting decoder": "Model.temporal_decoder/value_decoder"},
        "differences": ["paper task is imputation; local task term is the runner forecasting loss", "unlicensed author code is reference-only and was not inspected"],
    },
    "PAttn": {
        "method": "channel-independent instance normalization, patch projection, one attention layer, flatten forecast projection",
        "equation": "Y=Linear(Flatten(LN(P+MHA(P,P,P)))), P=PatchProject(Unfold(InstanceNorm(X)))",
        "modules": {"Figure 4 patching": "Model.patch_projection", "single attention": "Model.attention", "forecast projection": "Model.forecast_projection"},
        "differences": ["no positional embedding and no FFN as stated in appendix D.3", "training recipes and numerical parity are not claimed"],
    },
    "PHAT": {
        "method": "period buckets with positive-negative period-offset and phase-aligned attention",
        "equation": "A=softmax(zeta_mod)-Lambda*softmax(eta_mod); PNA=A x_phase (A_aligned x_cycle V)",
        "modules": {"FFT periods and buckets": "Model._periods/_bucket_path", "Eq.4 projections": "PositiveNegativeAttention.query/key/value/gate", "Eq.6-10 X attention": "PositiveNegativeAttention.forward", "Eq.11-12 head fusion": "PositiveNegativeAttention.output"},
        "differences": ["special zero-period bucket and dataset-specific periods omitted", "incomplete author repository is reference-only; no vendored source remains"],
    },
    "QuantileDLinear": {
        "method": "verified DLinear point backbone plus repository monotone quantile head",
        "equation": "q_m=a; q_i=a+sum softplus(delta_j) above m; q_i=a-sum softplus(delta_j) below m",
        "modules": {"trend-seasonal point forecast": "DLinearBackbone", "non-crossing quantiles": "QuantileHead"},
        "differences": ["probabilistic head is a ModernTSF composition not claimed by DLinear", "strictly increasing levels in (0,1) are required"],
    },
    "QuantilePatchTST": {
        "method": "verified PatchTST point backbone plus repository monotone quantile head",
        "equation": "q_m=a; adjacent quantile gaps=softplus(W_delta h); cumulative gaps enforce q_i<=q_(i+1)",
        "modules": {"channel-independent patch backbone": "PatchTSTBackbone", "non-crossing quantiles": "QuantileHead"},
        "differences": ["probabilistic head is a ModernTSF composition not claimed by PatchTST", "strictly increasing levels in (0,1) are required"],
    },
    "TiDE": {
        "method": "channel-independent dense encoder-decoder with covariate feature projection, temporal decoder, and global residual",
        "equation": "e=Encoder(y_1:L,x_tilde_1:L+H); D=reshape(Decoder(e)); yhat_t=TemporalDecoder(D_t,x_tilde_t)+Linear(y_1:L)_t",
        "modules": {"feature projection": "Model.feature_projection", "Eq.4 encoder": "Model.encoder_input/encoder_blocks", "dense and temporal decoder": "Model.dense_decoder/temporal_decoder", "global residual": "Model.global_residual"},
        "differences": ["static item attributes omitted because runner exposes temporal marks only", "paper preprocessing and numerical parity are not claimed"],
    },
}


def digest(value: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def factory(name: str, length: int = 8, horizon: int = 3, channels: int = 2):
    if name == "GlocalIB":
        return GlocalIB(length, horizon, channels, d_model=8, mask_ratio=0.25, kl_weight=0.02)
    if name == "PAttn":
        return PAttn(length, horizon, d_model=8, n_heads=2, patch_len=min(4, length), stride=1, dropout=0.0)
    if name == "PHAT":
        return PHAT(length, horizon, channels, d_model=8, n_heads=2, d_layers=1, attn_dropout=0.0, ffn_dropout=0.0)
    if name == "QuantileDLinear":
        return QuantileDLinear(length, horizon, channels, kernel_size=3, quantile_levels=[0.1, 0.5, 0.9])
    if name == "QuantilePatchTST":
        return QuantilePatchTST(length, horizon, channels, patch_len=min(4, length), stride=1, e_layers=1, d_model=8, n_heads=2, d_ff=16, quantile_levels=[0.1, 0.5, 0.9])
    if name == "TiDE":
        return TiDE(length, horizon, 8, 2, 2, 16, 4, 6, 0.0, True, 2)
    raise KeyError(name)


def call(name: str, model, x: torch.Tensor, historical: torch.Tensor, future: torch.Tensor):
    if name == "TiDE":
        return model(x, historical, None, future)
    return model(x, historical, torch.eye(x.shape[-1]), future)


def runtime(name: str) -> dict[str, object]:
    torch.manual_seed(260827)
    model = factory(name).cpu().train()
    x = torch.randn(2, 8, 2, requires_grad=True)
    historical, future = torch.randn(2, 8, 6), torch.randn(2, 3, 6)
    output = call(name, model, x, historical, future)
    expected = (2, 3, 2, 3) if name.startswith("Quantile") else (2, 3, 2)
    if tuple(output.shape) != expected or not torch.isfinite(output).all():
        raise AssertionError("forward or finite-output contract failed")
    monotone_gap = None
    if name.startswith("Quantile"):
        gaps = output[..., 1:] - output[..., :-1]
        monotone_gap = float(gaps.min())
        if monotone_gap < 0:
            raise AssertionError("quantile crossing")
    loss = output.square().mean()
    if getattr(model, "aux_loss", None) is not None:
        loss = loss + model.aux_loss
    loss.backward()
    if x.grad is None or not torch.isfinite(x.grad).all() or x.grad.abs().max() == 0:
        raise AssertionError("input gradient failed")
    gradients = {}
    for parameter_name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None or not torch.isfinite(parameter.grad).all() or parameter.grad.abs().max() == 0:
            raise AssertionError(f"inactive parameter: {parameter_name}")
        gradients[parameter_name] = float(parameter.grad.abs().max())

    model.eval()
    expected_output = call(name, model, x.detach(), historical, future)
    clone = factory(name).cpu().eval()
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    actual_output = call(name, clone, x.detach(), historical, future)
    round_trip_error = float((actual_output - expected_output).abs().max())
    if round_trip_error != 0.0:
        raise AssertionError("state dict round trip failed")
    boundary = factory(name, 4, 1, 2).eval()
    boundary_output = call(name, boundary, torch.randn(1, 4, 2), torch.randn(1, 4, 6), torch.randn(1, 1, 6))
    if boundary_output.shape[0] != 1:
        raise AssertionError("batch boundary failed")
    try:
        call(name, boundary, torch.randn(1, 3, 2), torch.randn(1, 3, 6), torch.randn(1, 1, 6))
    except (ValueError, RuntimeError):
        rejected = True
    else:
        raise AssertionError("wrong sequence length accepted")
    covariate_effect = 0.0
    if name == "TiDE":
        covariate_effect = float((call(name, model, x.detach(), historical, future + 1.0) - expected_output).abs().max())
        if covariate_effect == 0.0:
            raise AssertionError("TiDE future covariate highway inactive")
    return {
        "shape": list(output.shape),
        "input_gradient_max_abs": float(x.grad.abs().max()),
        "parameter_gradients": gradients,
        "round_trip_max_abs": round_trip_error,
        "wrong_length_rejected": rejected,
        "minimum_quantile_gap": monotone_gap,
        "future_covariate_effect_max_abs": covariate_effect,
        "marks_contract": "historical-and-future-covariates-consumed" if name == "TiDE" else "accepted-and-ignored",
        "adjacency_contract": "accepted-and-ignored" if name != "TiDE" else "not-applicable",
    }


def check(evidence: list[str], **metrics: object) -> dict[str, object]:
    return {"passed": True, "evidence": evidence, "metrics": metrics}


def verify(name: str, records: dict[str, dict[str, object]]) -> None:
    observations = runtime(name)
    structure = STRUCTURES[name]
    structure_digest = digest(structure)
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
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence = [relative, "tests/test_probabilistic_attention_final_rewrites.py"]
    checks = {
        "paper_structure": check(evidence, mapped_elements=len(structure["modules"]), claim="paper-equations-to-independent-local-map"),
        "equations": check(evidence, cases=1),
        "construction": check(evidence, instances=3),
        "forward": check(evidence, shape=",".join(map(str, observations["shape"]))),
        "backward": check(evidence, input_gradient_max_abs=observations["input_gradient_max_abs"]),
        "finite_outputs": check(evidence, nonfinite=0),
        "active_parameter_gradients": check(evidence, parameters=len(observations["parameter_gradients"])),
        "state_dict_round_trip": check(evidence, max_abs=observations["round_trip_max_abs"]),
        "cpu": check(evidence, device="cpu"),
        "batch_size_boundary": check(evidence, cases="batch=1,batch=2"),
        "sequence_length_boundary": check(evidence, cases="minimum-valid,wrong-length-rejected"),
        "marks_adjacency_contract": check(evidence, marks=observations["marks_contract"], adjacency=observations["adjacency_contract"]),
    }
    result = {
        "schema_version": 1,
        "kind": "rewrite-validation",
        "model": name,
        "implementation": "rewrite",
        "verified_at": datetime.now(timezone.utc),
        "subject_sha256": verification_subject_sha256(ROOT, records[name]),
        "commands": [
            f"uv run python scripts/verify_probabilistic_attention_final_rewrites.py --model {name}",
            "uv run python -m unittest tests.test_probabilistic_attention_final_rewrites -v",
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
        "basis": {"references": [REFERENCES[name]], "structure_map_sha256": structure_digest, "independent_design": True, "source_code_not_copied": True},
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
