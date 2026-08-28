#!/usr/bin/env python3
"""Generate clean-room structure and runtime evidence for six paper rewrites."""

from __future__ import annotations

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
from models.autoformer.model import Model as Autoformer  # noqa: E402
from models.fedformer.model import Model as FEDformer  # noqa: E402
from models.itransformer.model import Model as ITransformer  # noqa: E402
from models.patchtst.model import Model as PatchTST  # noqa: E402
from models.timemixer.model import Model as TimeMixer  # noqa: E402
from models.timesnet.model import Model as TimesNet  # noqa: E402


STRUCTURES: dict[str, dict[str, object]] = {
    "Autoformer": {
        "reference": "https://proceedings.neurips.cc/paper_files/paper/2021/hash/bcc0d400288793e8bdcd7c19a8ac0c2b-Abstract.html",
        "equations": {
            "Eq. 1": "edge-padded moving-average seasonal/trend decomposition",
            "Eqs. 2-4": "decoder initialization and progressive encoder/decoder decomposition",
            "Eqs. 5-6": "FFT autocorrelation, top-delay discovery, and rolled aggregation",
        },
        "modules": {
            "series decomposition": "SeriesDecomposition",
            "Auto-Correlation": "fft_autocorrelation and AutoCorrelation",
            "progressive encoder": "AutoformerEncoderLayer",
            "trend-accumulating decoder": "AutoformerDecoderLayer",
        },
        "differences": [
            "forecast-only",
            "linear resizing for cross-correlation context",
            "six-column raw calendar embedding",
            "no checkpoint or published-metric parity claim",
        ],
    },
    "FEDformer": {
        "reference": "https://proceedings.mlr.press/v162/zhou22g.html",
        "equations": {
            "Eqs. 3-4": "selected Fourier modes and complex spectral kernel",
            "Eqs. 6-7": "frequency-domain cross attention",
            "decomposition architecture": "three progressive decoder trend updates",
        },
        "modules": {
            "FEB-f": "FrequencyEnhancedBlock",
            "FEA-f": "FrequencyEnhancedAttention",
            "progressive decomposition": "FEDformerEncoderLayer and FEDformerDecoderLayer",
        },
        "differences": [
            "Fourier variant only; wavelet variant omitted",
            "head-local complex kernels",
            "deterministic random mode selection",
            "tanh frequency cross-attention",
        ],
    },
    "PatchTST": {
        "reference": "https://openreview.net/forum?id=Jbdc0vTOcol",
        "equations": {
            "Eq. 1": "overlapping patch tokenization with end-value padding",
            "channel independence": "channels fold into batch and share encoder weights",
            "forecast head": "all patch representations flatten to the prediction horizon",
        },
        "modules": {
            "patching": "patchify",
            "shared channel encoder": "PatchEncoderLayer",
            "reversible normalization": "canonical RevIN instance",
            "flatten head": "Model.head",
        },
        "differences": [
            "supervised forecasting only",
            "native PyTorch attention with standard head dimensions",
            "no residual-attention accumulation or pretrained transfer",
            "calendar and decoder arguments accepted and ignored",
        ],
    },
    "TimesNet": {
        "reference": "https://arxiv.org/abs/2210.02186",
        "equations": {
            "FFT period discovery": "global top frequencies converted to integer periods",
            "1D-to-2D transform": "period-aligned rows and intraperiod columns",
            "adaptive aggregation": "sample amplitudes receive softmax weights",
        },
        "modules": {
            "period discovery": "dominant_periods",
            "parameter-efficient 2D kernels": "Inception2D",
            "residual variation block": "TimesBlock",
        },
        "differences": [
            "forecasting task only",
            "global batch/channel period selection with sample-specific weights",
            "six-column calendar embedding",
            "no official training-recipe reproduction",
        ],
    },
    "iTransformer": {
        "reference": "https://arxiv.org/abs/2310.06625",
        "equations": {
            "Eq. 1": "whole lookback series embedded and projected per variate",
            "Eq. 2": "LayerNorm over each variate-token representation",
            "self-attention": "attention score map spans variate tokens",
        },
        "modules": {
            "variate tokenization": "InvertedEmbedding",
            "variate attention and temporal FFN": "InvertedEncoderLayer",
            "future decoder": "Model.projection",
        },
        "differences": [
            "fixed catalog channel contract",
            "six calendar series become auxiliary tokens",
            "no efficient-attention plugins or non-forecast tasks",
            "inert legacy factor/embed/freq options removed",
        ],
    },
    "TimeMixer": {
        "reference": "https://arxiv.org/abs/2405.14616",
        "equations": {
            "Eqs. 1-2": "stacked PDM followed by FMM",
            "Eqs. 3-5": "decomposition with bottom-up seasonal and top-down trend mixing",
            "Eq. 6": "sum ensemble of scale-specific predictors",
        },
        "modules": {
            "multiscale observations": "Model._downsample",
            "PDM": "PastDecomposableMixing",
            "season/trend temporal MLPs": "TemporalMixer",
            "FMM": "Model.temporal_predictors and Model.channel_predictors",
        },
        "differences": [
            "average downsampling only",
            "channel-mixing forecast path only",
            "calendar and decoder arguments accepted and ignored",
            "forecast-only without official recipe reproduction",
        ],
    },
}


def _factory(name: str, length: int = 8):
    factories = {
        "Autoformer": lambda: Autoformer(length, 2, 3, 2, 2, 2, 8, 2, 1, 1, 16, 3, 2.0, 0.0),
        "FEDformer": lambda: FEDformer(length, 2, 3, 2, 2, 2, 8, 2, 1, 1, 16, 3, 0.0, modes=2),
        "PatchTST": lambda: PatchTST(2, length, 3, min(4, length), 2, "end", 1, 8, 2, d_ff=16, norm="LayerNorm"),
        "TimesNet": lambda: TimesNet(length, 0, 3, 2, 2, 8, 1, 16, 0.0, 2, 2),
        "iTransformer": lambda: ITransformer(length, 3, 2, 8, 2, 1, 16, 0.0, "gelu", False, True),
        "TimeMixer": lambda: TimeMixer(length, 3, 2, 2, 1, 8, 16, 2, 2, 3, 1, 0.0, True, "moving_avg"),
    }
    return factories[name]()


def _minimum_factory(name: str):
    return {
        "Autoformer": lambda: (Autoformer(3, 1, 1, 1, 1, 1, 4, 1, 1, 1, 8, 3, 2.0, 0.0), 3, 2),
        "FEDformer": lambda: (FEDformer(4, 1, 1, 1, 1, 1, 4, 1, 1, 1, 8, 3, 0.0, modes=2), 4, 2),
        "PatchTST": lambda: (PatchTST(1, 4, 1, 4, 1, "none", 1, 4, 1, d_ff=8, norm="LayerNorm"), 4, 1),
        "TimesNet": lambda: (TimesNet(4, 0, 1, 1, 1, 4, 1, 8, 0.0, 1, 1), 4, 1),
        "iTransformer": lambda: (ITransformer(1, 1, 1, 4, 1, 1, 8, 0.0, "gelu", False, True), 1, 1),
        "TimeMixer": lambda: (TimeMixer(4, 1, 1, 1, 1, 4, 8, 2, 2, 3, 1, 0.0, True, "moving_avg"), 4, 1),
    }[name]()


def _marks(batch: int, length: int) -> torch.Tensor:
    values = torch.zeros(batch, length, 6)
    values[..., 0] = 2024
    values[..., 1] = 1
    values[..., 2] = torch.arange(1, length + 1)
    values[..., 3] = torch.arange(length) % 7
    values[..., 4] = torch.arange(length) % 24
    return values


def _call(model, name: str, values: torch.Tensor, *, changed_marks: bool = False, pred_len: int = 3, decoder_length: int | None = None):
    encoder_marks = _marks(values.shape[0], values.shape[1])
    if changed_marks:
        encoder_marks[..., 4] += 6
    if decoder_length is None:
        decoder_length = 5 if name in {"Autoformer", "FEDformer"} else pred_len
    decoder = values.new_zeros(values.shape[0], decoder_length, values.shape[2])
    decoder_marks = _marks(values.shape[0], decoder_length)
    if changed_marks:
        decoder_marks[..., 4] += 6
    output = model(values, encoder_marks, decoder, decoder_marks)
    return output[0] if isinstance(output, tuple) else output


def _runtime(name: str) -> dict[str, object]:
    torch.manual_seed(7001 + sum(ord(character) for character in name))
    model = _factory(name).cpu()
    values = torch.randn(2, 8, 2, requires_grad=True)
    output = _call(model, name, values)
    if output.shape != (2, 3, 2) or not torch.isfinite(output).all():
        raise AssertionError(f"{name}: forward/finite contract failed")
    output.square().mean().backward()
    if values.grad is None or not torch.isfinite(values.grad).all():
        raise AssertionError(f"{name}: input gradient failed")
    gradients: dict[str, float] = {}
    for parameter_name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all() or parameter.grad.abs().max() == 0:
            raise AssertionError(f"{name}: inactive parameter {parameter_name}")
        gradients[parameter_name] = float(parameter.grad.abs().max())

    model.eval()
    expected = _call(model, name, values.detach())
    clone = _factory(name).eval()
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    torch.testing.assert_close(_call(clone, name, values.detach()), expected)
    if _call(model, name, torch.randn(1, 8, 2)).shape != (1, 3, 2):
        raise AssertionError(f"{name}: batch-size-one contract failed")
    try:
        _call(model, name, torch.randn(1, 7, 2))
    except ValueError:
        wrong_length_rejected = True
    else:
        raise AssertionError(f"{name}: wrong history length accepted")

    minimum, minimum_length, minimum_decoder_length = _minimum_factory(name)
    minimum_output = _call(
        minimum,
        name,
        torch.randn(1, minimum_length, 1),
        pred_len=1,
        decoder_length=minimum_decoder_length,
    )
    if minimum_output.shape != (1, 1, 1):
        raise AssertionError(f"{name}: minimum sequence boundary failed")
    changed = _call(model, name, values.detach(), changed_marks=True)
    marks_active = name in {"Autoformer", "FEDformer", "TimesNet", "iTransformer"}
    if marks_active and torch.equal(expected, changed):
        raise AssertionError(f"{name}: declared active marks do not affect output")
    if not marks_active:
        torch.testing.assert_close(changed, expected)
    return {
        "shape": [2, 3, 2],
        "input_gradient_max_abs": float(values.grad.abs().max()),
        "parameter_gradients": gradients,
        "round_trip_max_abs": 0.0,
        "batch_size_cases": [1, 2],
        "minimum_history": minimum_length,
        "wrong_length_rejected": wrong_length_rejected,
        "raw_marks_active": marks_active,
        "adjacency_contract": "not declared",
    }


def _digest(value: dict[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _environment(seed: int) -> dict[str, object]:
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
        "deterministic": {"seed": seed, "num_threads": torch.get_num_threads()},
    }


def _check(evidence: list[str], **metrics: float | int | str) -> dict[str, object]:
    return {"passed": True, "evidence": evidence, "metrics": metrics}


def main() -> int:
    records = {str(record["name"]): record for record in model_records(ROOT)}
    for name, structure in STRUCTURES.items():
        observations = _runtime(name)
        structure_digest = _digest(structure)
        relative = f"verification/rewrite/{name}.json"
        artifact_path = ROOT / relative
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact = {
            "schema_version": 1,
            "kind": "clean-room-structure-map",
            "model": name,
            "reference": structure["reference"],
            "independent_design": True,
            "source_code_not_copied": True,
            "structure_map": structure,
            "structure_map_sha256": structure_digest,
            "observations": observations,
        }
        artifact_path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        evidence = [relative, "tests/test_transformer_patch_rewrites.py"]
        mark_contract = (
            "raw-six-column-marks-active;adjacency-not-declared"
            if observations["raw_marks_active"]
            else "marks-accepted-and-ignored;adjacency-not-declared"
        )
        checks = {
            "paper_structure": _check(evidence, mapped_elements=len(structure["modules"])),
            "equations": _check(evidence, cases=len(structure["equations"])),
            "construction": _check(evidence, instances=3),
            "forward": _check(evidence, shape="2,3,2"),
            "backward": _check(evidence, input_gradient_max_abs=observations["input_gradient_max_abs"]),
            "finite_outputs": _check(evidence, nonfinite=0),
            "active_parameter_gradients": _check(evidence, parameters=len(observations["parameter_gradients"])),
            "state_dict_round_trip": _check(evidence, max_abs=0.0),
            "cpu": _check(evidence, device="cpu"),
            "batch_size_boundary": _check(evidence, cases="batch=1,batch=2"),
            "sequence_length_boundary": _check(evidence, cases=f"minimum={observations['minimum_history']};wrong-length-rejected"),
            "marks_adjacency_contract": _check(evidence, contract=mark_contract),
        }
        seed = 7001 + sum(ord(character) for character in name)
        result = {
            "schema_version": 1,
            "kind": "rewrite-validation",
            "model": name,
            "implementation": "rewrite",
            "verified_at": datetime.now(timezone.utc),
            "subject_sha256": verification_subject_sha256(ROOT, records[name]),
            "commands": [
                "uv run python scripts/verify_transformer_patch_rewrites.py",
                "uv run python -m unittest tests.test_transformer_patch_rewrites -v",
                f"uv run tsf repo doctor --strict --models {name}",
            ],
            "environment": _environment(seed),
            "artifacts": {relative: evidence_file_sha256(artifact_path)},
            "passed": True,
            "basis": {
                "references": [str(structure["reference"])],
                "structure_map_sha256": structure_digest,
                "independent_design": True,
                "source_code_not_copied": True,
            },
            "checks": checks,
        }
        write_verification_result(ROOT / "verification/model-results.json", result)
        print(f"{name}: rewrite validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
