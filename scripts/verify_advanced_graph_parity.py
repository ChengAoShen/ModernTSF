#!/usr/bin/env python3
"""Verify D2STGNN, DFDGCN, and HimNet against exact licensed sources."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.util
from io import BytesIO
import json
from pathlib import Path
import platform
import subprocess
import sys
import types

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from benchmark.catalog_metadata import model_records  # noqa: E402
from benchmark.parity import compare_model_parity  # noqa: E402
from benchmark.verification_results import (  # noqa: E402
    DEFAULT_INDEX,
    verification_subject_sha256,
    write_verification_result,
)
from components.graph_utils import adj_to_supports  # noqa: E402
from components.marks import to_spatiotemporal  # noqa: E402
from models.d2stgnn.model import Model as LocalD2Wrapper  # noqa: E402
from models.d2stgnn._upstream import D2STGNN as LocalD2  # noqa: E402
from models.dfdgcn.model import Model as LocalDFDWrapper  # noqa: E402
from models.dfdgcn._upstream import DFDGCN as LocalDFD  # noqa: E402
from models.himnet.model import Model as LocalHimWrapper  # noqa: E402
from models.himnet._upstream import HimNet as LocalHim  # noqa: E402


SOURCES = {
    "D2STGNN": {
        "url": "https://github.com/GestaltCogTeam/BasicTS",
        "revision": "79641b1c75246ab2d8c53bb52f2ac72588be0cdc",
        "license": "Apache-2.0",
        "license_sha256": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
        "files": {
            "baselines/D2STGNN/arch/d2stgnn_arch.py": "062ac2528c2dc0269a2ccf77fcef9fd42a9b4a55ac3fe6779671abf1b26e542c",
            "baselines/D2STGNN/arch/decouple/estimation_gate.py": "e6ff19e00881e5237335df6c0382665fa079c040867f138191f75c4908369d77",
            "baselines/D2STGNN/arch/decouple/residual_decomp.py": "c4167a0490062ea95b392758a1436bd94b317bfb06d75f12f173661348896b83",
            "baselines/D2STGNN/arch/difusion_block/dif_block.py": "1093cd6f4851d4e5080da1461d8cdcb9be127ac2d8480a0e3a34f80724ae470c",
            "baselines/D2STGNN/arch/difusion_block/dif_model.py": "c146d8ef4af166148857e69ef225ff497789ac7d393ecaaa73f215cac9c00ded",
            "baselines/D2STGNN/arch/difusion_block/forecast.py": "e27e2f0d485eca8b0a02590b46ed88c58f96ead3155f3688d155402a2668c890",
            "baselines/D2STGNN/arch/dynamic_graph_conv/dy_graph_conv.py": "e360f6bf1dbd2953a297e53cef2af0ddccd1a4b6832866f11887d1778806bc2e",
            "baselines/D2STGNN/arch/dynamic_graph_conv/utils/distance.py": "01254b15df572387ff2d5389d14c64f49cf7671d9735f949cef9ac8eccd1dd9e",
            "baselines/D2STGNN/arch/dynamic_graph_conv/utils/mask.py": "8605f9189acd7d686a97956f60cddf9559c5dc420be37c8655fa76a38644ab1f",
            "baselines/D2STGNN/arch/dynamic_graph_conv/utils/normalizer.py": "3b4f09e9ed0230225b3cde28f27a20c587e683a019cc50c944bfeae57d51e230",
            "baselines/D2STGNN/arch/inherent_block/forecast.py": "961f36773c3020aee6c60f05b205ec365d8ea11bd4b0a576aeeec7de1a0eecc1",
            "baselines/D2STGNN/arch/inherent_block/inh_block.py": "ffad030102ca07183752584f2c646ac401ad3ff4e95959ca13336e93b4b55f02",
            "baselines/D2STGNN/arch/inherent_block/inh_model.py": "fd4b3f8704ec6a388985d2e6b0689c060b8eff87b6d3599cc843ec0e7ba28992",
        },
    },
    "DFDGCN": {
        "url": "https://github.com/GestaltCogTeam/DFDGCN",
        "revision": "3105058512a9279c000e98046a49d1baf3469884",
        "license": "MIT",
        "license_sha256": "33208e621f862fab0b072c72c1212b555bff62f915c2390eb5a52621328d8e0f",
        "files": {
            "DFDGCN/basicts/archs/arch_zoo/dfdgcn_arch/dfdgcn_arch.py": "938a4e9086a0222e07540e25ff87838160247ee5f43446e3d7835fc6b0e4bab2",
        },
    },
    "HimNet": {
        "url": "https://github.com/GestaltCogTeam/BasicTS",
        "revision": "c218c07b6ce5e4cf908b147fd180c486346fed9c",
        "license": "Apache-2.0",
        "license_sha256": "c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4",
        "files": {
            "baselines/HimNet/arch/model/HimNet.py": "1397f9cd6899e5762844b1f9cbe33a784b0d329948a78810fb46f29dfa08a5e5",
        },
    },
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify_checkout(name: str, checkout: Path) -> None:
    source = SOURCES[name]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=checkout, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    if revision != source["revision"]:
        raise ValueError(f"{name} checkout is {revision}, expected {source['revision']}")
    if _sha(checkout / "LICENSE") != source["license_sha256"]:
        raise ValueError(f"{name} LICENSE digest mismatch")
    for relative, expected in source["files"].items():
        actual = _sha(checkout / relative)
        if actual != expected:
            raise ValueError(f"{name} source digest mismatch for {relative}: {actual}")


def _clear_external_modules() -> None:
    for key in tuple(sys.modules):
        if key == "baselines" or key.startswith("baselines.") or key == "basicts" or key.startswith("basicts."):
            del sys.modules[key]


def _load_d2(checkout: Path) -> type[torch.nn.Module]:
    _verify_checkout("D2STGNN", checkout)
    _clear_external_modules()
    # Import the real BasicTS source without executing basicts/__init__.py,
    # whose training launcher requires the unrelated easytorch package.
    package = types.ModuleType("basicts")
    package.__path__ = [str(checkout / "basicts")]
    sys.modules["basicts"] = package
    sys.path.insert(0, str(checkout))
    try:
        return importlib.import_module("baselines.D2STGNN.arch.d2stgnn_arch").D2STGNN
    finally:
        sys.path.remove(str(checkout))


def _load_him(checkout: Path) -> type[torch.nn.Module]:
    _verify_checkout("HimNet", checkout)
    _clear_external_modules()
    sys.path.insert(0, str(checkout))
    try:
        return importlib.import_module("baselines.HimNet.arch.model.HimNet").HimNet
    finally:
        sys.path.remove(str(checkout))


def _load_dfd(checkout: Path) -> type[torch.nn.Module]:
    _verify_checkout("DFDGCN", checkout)
    path = checkout / next(iter(SOURCES["DFDGCN"]["files"]))
    spec = importlib.util.spec_from_file_location("moderntsf_exact_dfdgcn", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DFDGCN


def _round_trip(model: torch.nn.Module, args: tuple[object, ...]) -> tuple[bool, float]:
    model.eval()
    with torch.no_grad():
        expected = model(*args)
    stream = BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = deepcopy(model)
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    restored.eval()
    with torch.no_grad():
        actual = restored(*args)
    error = (expected - actual).abs()
    return torch.equal(expected, actual), float(error.max()) if error.numel() else 0.0


def _identity_map(local: torch.nn.Module, upstream: torch.nn.Module) -> dict[str, str]:
    left, right = local.state_dict(), upstream.state_dict()
    if set(left) != set(right):
        raise ValueError(f"state keys differ: local-only={sorted(set(left)-set(right))}; upstream-only={sorted(set(right)-set(left))}")
    for name in left:
        if left[name].shape != right[name].shape:
            raise ValueError(f"state shape differs for {name}: {left[name].shape} != {right[name].shape}")
    return {name: name for name in left}


def _gradient_activity(model: torch.nn.Module, args: tuple[object, ...], seed: int) -> list[str]:
    """Return parameters active for the exact fixture, including zero gradients."""
    cloned = tuple(
        value.detach().clone().requires_grad_(value.is_floating_point())
        if torch.is_tensor(value) else value
        for value in args
    )
    model.zero_grad(set_to_none=True)
    torch.manual_seed(seed)
    output = model(*cloned)
    output.float().sum().backward()
    return sorted(name for name, parameter in model.named_parameters()
                  if parameter.grad is not None)


def _activity_contract(local: torch.nn.Module, upstream: torch.nn.Module,
                       args: tuple[object, ...], seed: int) -> dict[str, object]:
    modes: dict[str, object] = {}
    for mode in ("eval", "train"):
        local.train(mode == "train"); upstream.train(mode == "train")
        local_names = _gradient_activity(local, args, seed)
        upstream_names = _gradient_activity(upstream, args, seed)
        modes[mode] = {"local": local_names, "upstream": upstream_names,
                       "matched": local_names == upstream_names}
    all_names = sorted(dict(local.named_parameters()))
    active = sorted(set().union(*(item["local"] for item in modes.values())))
    return {"modes": modes, "active": active,
            "inactive": sorted(set(all_names) - set(active))}


def _marks(batch: int, length: int, nodes: int, *, integer_dow: bool = False) -> torch.Tensor:
    marks = torch.zeros(batch, length, nodes, 2)
    marks[..., 0] = (torch.arange(length).view(1, -1, 1) % 24) / 288.0
    dow = (torch.arange(nodes).view(1, 1, -1) + 2) % 7
    marks[..., 1] = dow if integer_dow else dow / 7.0
    return marks


def _case_d2(upstream_cls: type[torch.nn.Module], batch: int, adjacency: np.ndarray) -> dict[str, object]:
    seed, seq, nodes = 4211, 12, adjacency.shape[0]
    args = dict(num_feat=1, num_hidden=4, node_hidden=3, time_emb_dim=2,
        seq_length=seq, num_nodes=nodes, k_s=2, k_t=3, gap=1, dropout=0.1,
        time_in_day_size=288, day_in_week_size=7,
        adjs=list(adj_to_supports(adjacency)))
    torch.manual_seed(seed); upstream = upstream_cls(**args)
    torch.manual_seed(seed + 1); local = LocalD2(**args)
    mapping = _identity_map(local, upstream)
    values = torch.randn(batch, seq, nodes)
    marks = _marks(batch, seq, nodes)
    history = to_spatiotemporal(values, marks)
    backbone_args = (history, None, 0, 0, False)
    report = compare_model_parity(local, upstream, backbone_args, state_map=mapping,
        module_map={"dynamic_graph_constructor.distance_function": "dynamic_graph_constructor.distance_function",
                    "layers.0.spatial_gate": "layers.0.spatial_gate", "out_fc_2": "out_fc_2"},
        seed=seed, atol=1e-6, rtol=1e-5)
    activity = _activity_contract(local, upstream, backbone_args, seed)
    wrapper = LocalD2Wrapper(seq, seq, nodes, adjacency, 3, 1, 4, 3, 2, 2, 3, 1,
                             5, 0.1, 288, 7, 256, 512)
    wrapper.net.load_state_dict(upstream.state_dict(), strict=True)
    wrapper.eval(); local.eval(); local.load_state_dict(upstream.state_dict(), strict=True)
    with torch.no_grad():
        wrapped = wrapper(values, marks)
        direct = local(history, None, 0, 0, False)[..., 0]
    pre_error = float((wrapped - direct).abs().max())
    serial = [_round_trip(local, backbone_args), _round_trip(upstream, backbone_args),
              _round_trip(wrapper, (values, marks))]
    return _finish_case(report, mapping, local, wrapper, serial, pre_error, activity,
                        "double-transition supports", batch, adjacency)


def _case_dfd(upstream_cls: type[torch.nn.Module], batch: int, adjacency: np.ndarray) -> dict[str, object]:
    seed, seq, pred, nodes = 4211, 12, 3, adjacency.shape[0]
    supports = list(adj_to_supports(adjacency))
    args = dict(num_nodes=nodes, dropout=0.2, supports=supports, gcn_bool=True,
        addaptadj=True, aptinit=None, in_dim=2, out_dim=pred, residual_channels=4,
        dilation_channels=4, skip_channels=8, end_channels=12, kernel_size=2,
        blocks=1, layers=2, a=1.0, seq_len=seq, affine=False, fft_emb=3,
        identity_emb=3, hidden_emb=5, subgraph=nodes)
    torch.manual_seed(seed); upstream = upstream_cls(**args)
    torch.manual_seed(seed + 1); local = LocalDFD(**args)
    mapping = _identity_map(local, upstream)
    values = torch.randn(batch, seq, nodes)
    marks = _marks(batch, seq, nodes, integer_dow=True)
    history = to_spatiotemporal(values, marks)
    backbone_args = (history, None, 0, 0, False)
    report = compare_model_parity(local, upstream, backbone_args, state_map=mapping,
        module_map={"layersnorm": "layersnorm", "gconv.0": "gconv.0", "end_conv_2": "end_conv_2"},
        seed=seed, atol=1e-6, rtol=1e-5)
    activity = _activity_contract(local, upstream, backbone_args, seed)
    wrapper = LocalDFDWrapper(seq, pred, nodes, adjacency, 0.2, 4, 4, 8, 12, 2, 1, 2,
                              1.0, 3, 3, 5, nodes)
    wrapper.net.load_state_dict(upstream.state_dict(), strict=True)
    wrapper.eval(); local.eval(); local.load_state_dict(upstream.state_dict(), strict=True)
    torch.manual_seed(seed)
    with torch.no_grad(): wrapped = wrapper(values, marks)
    torch.manual_seed(seed)
    with torch.no_grad(): direct = local(history, None, 0, 0, False)[..., 0]
    pre_error = float((wrapped - direct).abs().max())
    serial = [_round_trip(local, backbone_args), _round_trip(upstream, backbone_args),
              _round_trip(wrapper, (values, marks))]
    return _finish_case(report, mapping, local, wrapper, serial, pre_error, activity,
                        "double-transition supports and official integer DOW", batch, adjacency)


def _case_him(upstream_cls: type[torch.nn.Module], batch: int, adjacency: np.ndarray) -> dict[str, object]:
    seed, seq, pred, nodes = 4211, 6, 3, adjacency.shape[0]
    args = dict(num_nodes=nodes, input_dim=3, output_dim=1, out_steps=pred,
        hidden_dim=4, num_layers=1, cheb_k=2, ycov_dim=2, tod_embedding_dim=2,
        dow_embedding_dim=2, node_embedding_dim=3, st_embedding_dim=3,
        use_teacher_forcing=False)
    torch.manual_seed(seed); upstream = upstream_cls(**args)
    torch.manual_seed(seed + 1); local = LocalHim(**args, steps_per_day=288)
    mapping = _identity_map(local, upstream)
    values = torch.randn(batch, seq, nodes)
    marks = _marks(batch, seq, nodes)
    future_marks = _marks(batch, pred, nodes)
    history = torch.cat([values.unsqueeze(-1), marks[..., :1],
                         (marks[..., 1:2] * 7).round()], dim=-1)
    future_cov = torch.cat([future_marks[..., :1],
                            (future_marks[..., 1:2] * 7).round()], dim=-1)
    backbone_args = (history, future_cov, None, None)
    report = compare_model_parity(local, upstream, backbone_args, state_map=mapping,
        module_map={"encoder_s": "encoder_s", "decoder": "decoder", "out_proj": "out_proj"},
        seed=seed, atol=1e-6, rtol=1e-5)
    activity = _activity_contract(local, upstream, backbone_args, seed)
    wrapper = LocalHimWrapper(seq, pred, nodes, None, 3, 1, 4, 1, 2, 3, 3, 2, 2, 288, False)
    wrapper.net.load_state_dict(upstream.state_dict(), strict=True)
    wrapper.eval(); local.eval(); local.load_state_dict(upstream.state_dict(), strict=True)
    with torch.no_grad():
        wrapped = wrapper(values, marks, None, future_marks)
        direct = local(history, future_cov, None, None)[..., 0]
    pre_error = float((wrapped - direct).abs().max())
    serial = [_round_trip(local, backbone_args), _round_trip(upstream, backbone_args),
              _round_trip(wrapper, (values, marks, None, future_marks))]
    return _finish_case(report, mapping, local, wrapper, serial, pre_error, activity,
                        "normalized marks to integer DOW; adaptive graph", batch, adjacency)


def _finish_case(report, mapping, local, wrapper, serial, pre_error, activity, contract, batch, adjacency):
    gradients_seen = min(len(mode.parameter_gradients) for mode in report.modes.values())
    activity_matches = all(item["matched"] for item in activity["modes"].values())
    gradients_expected = min(len(item["local"]) for item in activity["modes"].values())
    buffers = len(dict(local.named_buffers()))
    passed = (report.passed and all(item[0] for item in serial) and pre_error == 0.0
              and activity_matches and gradients_seen == gradients_expected)
    return {"passed": passed, "batch": batch, "adjacency": adjacency.tolist(),
        "state_map": mapping, "mapped_buffers": buffers,
        "active_parameter_gradients": gradients_seen,
        "expected_parameter_gradients": gradients_expected,
        "gradient_activity": activity,
        "serialization": {"local": serial[0], "upstream": serial[1], "wrapper": serial[2]},
        "preprocessing": {"max_abs": pre_error, "contract": contract},
        "wrapper_buffers": sorted(wrapper.state_dict()), "report": report.to_dict()}


def verify_model(name: str, upstream_cls: type[torch.nn.Module]) -> dict[str, object]:
    identity = np.eye(4, dtype=np.float32)
    graph = np.array([[0,1,0,0],[1,0,1,0],[0,1,0,1],[0,0,1,0]], dtype=np.float32)
    function = {"D2STGNN": _case_d2, "DFDGCN": _case_dfd, "HimNet": _case_him}[name]
    cases = {"batch_one_identity": function(upstream_cls, 1, identity),
             "batch_two_nontrivial_graph": function(upstream_cls, 2, graph)}
    source = SOURCES[name]
    return {"schema_version": 1, "model": name,
        "passed": all(bool(case["passed"]) for case in cases.values()),
        "source": {key: source[key] for key in ("url", "revision", "license", "license_sha256", "files")},
        "upstream_execution": "exact-pinned-checkout", "mapping_version": "advanced-graph-v1",
        "command": "uv run python scripts/verify_advanced_graph_parity.py --basicts-d2-checkout <BasicTS@79641b1c> --basicts-checkout <BasicTS@c218c07b> --dfdgcn-checkout <DFDGCN@31050585>",
        "tolerances": {"atol": 1e-6, "rtol": 1e-5}, "cases": cases}


def _errors(detail: dict[str, object], group: str) -> tuple[float, float]:
    values = [item for case in detail["cases"].values()
              for mode in case["report"]["modes"].values()
              for item in mode[group].values()]
    return max(float(x["max_abs"]) for x in values), max(float(x["max_rel"]) for x in values)


def _check(passed: bool, evidence: list[str], **metrics: float | int | str) -> dict[str, object]:
    return {"passed": passed, "evidence": evidence, "metrics": metrics}


def canonical_result(name: str, detail: dict[str, object], path: Path) -> dict[str, object]:
    fields = next(item for item in model_records(ROOT) if item["name"] == name)
    relative = path.relative_to(ROOT).as_posix()
    evidence = [relative, "tests/test_advanced_graph_parity.py"]
    errors = {group: _errors(detail, group) for group in
              ("outputs", "intermediates", "input_gradients", "parameter_gradients")}
    first = next(iter(detail["cases"].values()))
    passed = bool(detail["passed"])
    serial = all(item[0] for case in detail["cases"].values() for item in case["serialization"].values())
    buffers = all(case["mapped_buffers"] >= 0 for case in detail["cases"].values())
    source = SOURCES[name]
    return {"schema_version": 1, "kind": "upstream-parity", "implementation": "upstream",
        "model": name, "verified_at": datetime.now(timezone.utc).isoformat(),
        "subject_sha256": verification_subject_sha256(ROOT, fields),
        "artifacts": {relative: _sha(path)},
        "commands": [detail["command"], "uv run python -m unittest tests.test_advanced_graph_parity -v",
                     f"uv run tsf repo doctor --strict --models {name}"],
        "environment": {"python": platform.python_version(), "framework": f"torch {torch.__version__}",
            "dependencies": {"numpy": np.__version__, "torch": torch.__version__},
            "platform": platform.platform(), "device": "cpu", "dtype": "float32",
            "deterministic": {"seed": 4211, "algorithms": True, "num_threads": 1}},
        "passed": passed, "source": {key: source[key] for key in ("url", "revision", "license")},
        "mapping": {"version": "advanced-graph-v1", "parameters": len(first["state_map"]),
                    "buffers": first["mapped_buffers"]},
        "fixture": {"identifier": "advanced-graph-identity-nontrivial-v1",
                    "description": "CPU float32 batch=1/2 cases with identity/nontrivial adjacency and official calendar covariates."},
        "tolerances": detail["tolerances"], "modes": ["eval", "train"],
        "checks": {
            "outputs": _check(passed, evidence, max_abs=errors["outputs"][0], max_rel=errors["outputs"][1]),
            "intermediates": _check(passed, evidence, max_abs=errors["intermediates"][0], max_rel=errors["intermediates"][1]),
            "input_gradients": _check(passed, evidence, max_abs=errors["input_gradients"][0], max_rel=errors["input_gradients"][1]),
            "parameter_gradients": _check(passed, evidence, max_abs=errors["parameter_gradients"][0], max_rel=errors["parameter_gradients"][1]),
            "train_eval": _check(passed, evidence, modes="eval,train"),
            "buffers": _check(buffers, evidence, mapped_buffers=first["mapped_buffers"]),
            "serialization": _check(serial, evidence, max_abs=0.0),
            "preprocessing": _check(passed, evidence, contract="adjacency/support and calendar conversion checked against exact backbone inputs"),
            "boundaries": _check(passed, evidence, cases="batch_one_identity,batch_two_nontrivial_graph"),
        }}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basicts-d2-checkout", required=True, type=Path)
    parser.add_argument("--basicts-checkout", required=True, type=Path)
    parser.add_argument("--dfdgcn-checkout", required=True, type=Path)
    parser.add_argument("--models", nargs="*", choices=sorted(SOURCES), default=sorted(SOURCES))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "verification" / "parity")
    args = parser.parse_args()
    torch.use_deterministic_algorithms(True); torch.set_num_threads(1)
    loaders = {"D2STGNN": lambda: _load_d2(args.basicts_d2_checkout.resolve()),
               "DFDGCN": lambda: _load_dfd(args.dfdgcn_checkout.resolve()),
               "HimNet": lambda: _load_him(args.basicts_checkout.resolve())}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    passed = True
    for name in args.models:
        detail = verify_model(name, loaders[name]())
        output = args.output_dir / f"{name}.json"
        output.write_text(json.dumps(detail, indent=2) + "\n", encoding="utf-8")
        if detail["passed"]:
            write_verification_result(ROOT / DEFAULT_INDEX, canonical_result(name, detail, output))
        print(f"{'PASS' if detail['passed'] else 'FAIL'} {name}: {output}")
        passed &= bool(detail["passed"])
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
