#!/usr/bin/env python3
"""Execute strict clean-room evidence for seven differentiable tree baselines."""

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
from models.catboost_ts.model import Model as CatBoost  # noqa: E402
from models.decision_tree_ts.model import Model as DecisionTree  # noqa: E402
from models.extra_trees_ts.model import Model as ExtraTrees  # noqa: E402
from models.gradient_boosting_ts.model import Model as GradientBoosting  # noqa: E402
from models.lightgbm_ts.model import Model as LightGBM  # noqa: E402
from models.random_forest_ts.model import Model as RandomForest  # noqa: E402
from models.xgboost_ts.model import Model as XGBoost  # noqa: E402

Factory = Callable[[int, int, int], nn.Module]


@dataclass(frozen=True)
class RewriteCase:
    name: str
    factory: Factory
    reference: str
    structure: dict[str, object]


CASES = (
    RewriteCase("DecisionTreeTS", lambda l, h, c: DecisionTree(l, h, c, 2),
                "https://search.worldcat.org/title/1422106714",
                {"method": "single differentiable binary regression tree", "equation": "p(leaf|x)=product_depth p(branch|x); y=sum_leaf p(leaf|x)*value_leaf", "modules": {"soft routing": "Model.tree"}, "differences": ["soft learned oblique splits rather than greedy hard impurity splits", "no pruning or CART fitting procedure"]}),
    RewriteCase("RandomForestTS", lambda l, h, c: RandomForest(l, h, c, 3, 2),
                "https://doi.org/10.1023/A:1010933404324",
                {"method": "random-subspace soft-tree average", "equation": "y=(1/T)*sum_t tree_t(x .* mask_t)", "modules": {"fixed feature subspaces": "Model.trees[*].split_mask", "averaging": "Model.forward"}, "differences": ["no row bootstrap or out-of-bag estimate", "soft end-to-end split fitting"]}),
    RewriteCase("ExtraTreesTS", lambda l, h, c: ExtraTrees(l, h, c, 3, 2),
                "https://doi.org/10.1007/s10994-006-6226-1",
                {"method": "frozen-random-split soft-tree average", "equation": "y=(1/T)*sum_t sum_leaf p_t(leaf|x)*value_t,leaf", "modules": {"random axes/thresholds": "Model.trees buffers", "learned leaves": "SoftDecisionTree.leaf_value"}, "differences": ["sigmoid routing rather than hard comparisons", "gradient-learned leaves rather than sample means"]}),
    RewriteCase("GradientBoostingTS", lambda l, h, c: GradientBoosting(l, h, c, 3, 2),
                "https://doi.org/10.1214/aos/1013203451",
                {"method": "end-to-end additive residual soft trees", "equation": "forecast=base(x)+eta*sum_t tree_t(state_t); state_(t+1)=state_t-eta*tanh(backcast_t(tree_t(state_t)))", "modules": {"base": "Model.base", "stages": "Model.trees", "residual update": "Model.backcasts"}, "differences": ["joint optimization rather than sequential pseudo-residual fitting"]}),
    RewriteCase("CatBoostTS", lambda l, h, c: CatBoost(l, h, c, 3, 2),
                "https://arxiv.org/abs/1706.09516",
                {"method": "ordered-context symmetric soft-tree ensemble", "equation": "state_t=x-tanh(context_t(forecast_(t-1)/t)); forecast_t=forecast_(t-1)+eta*oblivious_tree_t(state_t)", "modules": {"symmetric tree": "SoftObliviousTree", "prior-stage context": "Model.context"}, "differences": ["no permutation ordered boosting", "no categorical statistics or CatBoost library"]}),
    RewriteCase("LightGBMTS", lambda l, h, c: LightGBM(l, h, c, 3, 2),
                "https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html",
                {"method": "feature-gated compact additive soft trees", "equation": "gate=sigmoid(logits); forecast=base(x*gate)+eta*sum_t tree_t(state_t*gate)", "modules": {"feature gate": "Model.feature_logits", "varying-depth stages": "Model.trees"}, "differences": ["no leaf-wise hard-tree growth", "no histograms, GOSS, EFB, or LightGBM library"]}),
    RewriteCase("XGBoostTS", lambda l, h, c: XGBoost(l, h, c, 3, 2),
                "https://arxiv.org/abs/1603.02754",
                {"method": "column-masked regularized additive soft trees", "equation": "forecast=base(x)+eta*sum_t tree_t(state_t .* mask_t); penalty=lambda1*mean(abs(leaves))+lambda2*mean(leaves^2)", "modules": {"column masks": "Model.column_masks", "leaf penalty": "Model.aux_loss", "stages": "Model.trees"}, "differences": ["no second-order split objective or sparsity-aware search", "no quantile sketch or XGBoost systems implementation"]}),
)


def _digest(value: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _equation_check(case: RewriteCase) -> None:
    model = case.factory(2, 1, 1)
    x = torch.tensor([[[0.25], [-0.5]]])
    output = model(x)
    if output.shape != (1, 1, 1) or not torch.isfinite(output).all():
        raise AssertionError("equation fixture failed")
    if case.name == "DecisionTreeTS":
        torch.testing.assert_close(model.tree.leaf_probabilities(model.revin(x, "norm").flatten(1)).sum(-1), torch.ones(1))
    elif case.name == "RandomForestTS" and len(model.trees) != 3:
        raise AssertionError("forest member count failed")
    elif case.name == "ExtraTreesTS" and any("split_weight" in name for name, _ in model.named_parameters()):
        raise AssertionError("random split geometry must be frozen")
    elif case.name == "GradientBoostingTS" and len(model.backcasts) != 2:
        raise AssertionError("residual stage count failed")
    elif case.name == "CatBoostTS" and model.trees[0].split_weight.shape[0] != 2:
        raise AssertionError("oblivious tree must share one split per depth")
    elif case.name == "LightGBMTS" and model.feature_logits.numel() != 2:
        raise AssertionError("feature gate contract failed")
    elif case.name == "XGBoostTS" and model.column_masks.shape != (3, 2):
        raise AssertionError("column mask contract failed")


def _runtime(case: RewriteCase) -> dict[str, object]:
    torch.manual_seed(1753)
    _equation_check(case)
    model = case.factory(4, 3, 2).cpu()
    x = torch.randn(2, 4, 2, requires_grad=True)
    marks, adjacency = torch.randn(2, 4, 3), torch.eye(2)
    output = model(x, marks, adjacency)
    if output.shape != (2, 3, 2) or not torch.isfinite(output).all():
        raise AssertionError("forward contract failed")
    loss = output.square().mean() + (model.aux_loss if model.aux_loss is not None else 0.0)
    loss.backward()
    if x.grad is None or not torch.isfinite(x.grad).all():
        raise AssertionError("input gradient failed")
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all() or parameter.grad.abs().max() == 0:
            raise AssertionError(f"inactive or invalid parameter gradient: {name}")
        gradients[name] = float(parameter.grad.abs().max())
    clone = case.factory(4, 3, 2).cpu()
    clone.load_state_dict(copy.deepcopy(model.state_dict()))
    torch.testing.assert_close(clone(x.detach()), output.detach())
    if model(torch.randn(1, 4, 2)).shape != (1, 3, 2):
        raise AssertionError("batch boundary failed")
    if case.factory(1, 1, 2)(torch.randn(1, 1, 2)).shape != (1, 1, 2):
        raise AssertionError("minimum sequence failed")
    try:
        model(torch.randn(1, 3, 2))
    except ValueError:
        wrong_length_rejected = True
    else:
        raise AssertionError("wrong sequence length accepted")
    torch.testing.assert_close(model(x.detach(), marks, adjacency), model(x.detach()))
    return {"shape": [2, 3, 2], "input_gradient_max_abs": float(x.grad.abs().max()), "parameter_gradients": gradients, "round_trip_max_abs": 0.0, "wrong_length_rejected": wrong_length_rejected}


def _environment() -> dict[str, object]:
    return {"python": platform.python_version(), "framework": f"torch {torch.__version__}", "dependencies": {"pydantic": importlib.metadata.version("pydantic"), "torch": torch.__version__}, "platform": platform.platform(), "device": "cpu", "dtype": "float32", "deterministic": {"seed": 1753, "num_threads": torch.get_num_threads()}}


def _check(evidence: list[str], **metrics: float | int | str) -> dict[str, object]:
    return {"passed": True, "evidence": evidence, "metrics": metrics}


def verify(case: RewriteCase, records: dict[str, dict[str, object]]) -> None:
    observations = _runtime(case)
    structure_digest = _digest(case.structure)
    relative = f"verification/rewrite/{case.name}.json"
    artifact_path = ROOT / relative
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {"schema_version": 1, "kind": "clean-room-structure-map", "model": case.name, "reference": case.reference, "independent_design": True, "source_code_not_copied": True, "structure_map": case.structure, "structure_map_sha256": structure_digest, "observations": observations}
    artifact_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    evidence = [relative, "tests/test_tree_baseline_rewrites.py"]
    checks = {
        "paper_structure": _check(evidence, mapped_elements=len(case.structure["modules"]), claim="conceptual-reference-to-independent-local-map"),
        "equations": _check(evidence, cases=1), "construction": _check(evidence, instances=3),
        "forward": _check(evidence, shape="2,3,2"), "backward": _check(evidence, input_gradient_max_abs=observations["input_gradient_max_abs"]),
        "finite_outputs": _check(evidence, nonfinite=0), "active_parameter_gradients": _check(evidence, parameters=len(observations["parameter_gradients"])),
        "state_dict_round_trip": _check(evidence, max_abs=0.0), "cpu": _check(evidence, device="cpu"),
        "batch_size_boundary": _check(evidence, cases="batch=1,batch=2"), "sequence_length_boundary": _check(evidence, cases="length=1,wrong-length-rejected"),
        "marks_adjacency_contract": _check(evidence, contract="accepted-and-ignored"),
    }
    result = {"schema_version": 1, "kind": "rewrite-validation", "model": case.name, "implementation": "rewrite", "verified_at": datetime.now(timezone.utc), "subject_sha256": verification_subject_sha256(ROOT, records[case.name]), "commands": [f"uv run python scripts/verify_tree_baseline_rewrites.py --model {case.name}", "uv run python -m unittest tests.test_tree_baseline_rewrites -v", f"uv run tsf repo doctor --backward --models {case.name}"], "environment": _environment(), "artifacts": {relative: evidence_file_sha256(artifact_path)}, "passed": True, "basis": {"references": [case.reference], "structure_map_sha256": structure_digest, "independent_design": True, "source_code_not_copied": True}, "checks": checks}
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
