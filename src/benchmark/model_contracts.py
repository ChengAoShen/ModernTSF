"""Executable construction and forward contracts for every catalog model."""

from __future__ import annotations

import contextlib
import gc
import io
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch

from benchmark.registry.models import MODEL_CATALOG, ModelSpec


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ContractFailure:
    model: str
    stage: str
    error: str


def _task_for(spec: ModelSpec) -> SimpleNamespace:
    values: dict[str, int | str] = {
        "seq_len": 96,
        "label_len": 48,
        "pred_len": 12,
        "features": "M",
    }
    values.update(spec.contract_task)
    return SimpleNamespace(**values)


def _params_for(spec: ModelSpec) -> dict:
    config = tomllib.loads((ROOT / spec.config_path).read_text(encoding="utf-8"))
    return dict(config["model"].get("params", {}))


def _forward_contract(
    model, spec: ModelSpec, task, params: dict, *, backward: bool = False
) -> None:
    batch = 2
    channels = int(params.get("enc_in", params.get("num_nodes", 1)))
    x = torch.randn(batch, task.seq_len, channels)
    x_mark = torch.zeros(batch, task.seq_len, 6)
    dec = torch.zeros(batch, task.label_len + task.pred_len, channels)
    dec_mark = torch.zeros(batch, task.label_len + task.pred_len, 6)
    model.eval()
    grad_context = contextlib.nullcontext() if backward else torch.no_grad()
    with grad_context:
        try:
            output = model(x, x_mark, dec, dec_mark)
        except TypeError:
            output = model(x)
    if isinstance(output, tuple):
        output = output[0]
    if not torch.is_tensor(output):
        raise TypeError(f"forward returned {type(output).__name__}, expected Tensor")
    if output.shape[0] != batch or output.shape[1] != task.pred_len:
        raise ValueError(
            f"output shape {tuple(output.shape)} violates (B, pred_len, ...) contract"
        )
    if not torch.isfinite(output).all():
        raise ValueError("forward output contains NaN or Inf")
    output_type = getattr(model, "output_type", spec.output_type)
    if output_type != spec.output_type:
        raise ValueError(
            f"model output_type={output_type!r} disagrees with ModelSpec "
            f"output_type={spec.output_type!r}"
        )
    if output_type == "point" and output.ndim != 3:
        raise ValueError(f"point model returned rank-{output.ndim} output")
    if output_type == "quantile" and output.ndim != 4:
        raise ValueError(f"quantile model returned rank-{output.ndim} output")
    if output_type == "distribution":
        if output.ndim != 4 or output.shape[-1] != 2:
            raise ValueError(
                f"distribution model must return (B, T, C, 2), got {tuple(output.shape)}"
            )
    if backward:
        if not output.requires_grad:
            raise ValueError("forward output is detached; training cannot backpropagate")
        output.float().mean().backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        if not gradients:
            raise ValueError("backward produced no parameter gradients")
        if not all(torch.isfinite(gradient).all() for gradient in gradients):
            raise ValueError("backward produced NaN or Inf gradients")


def audit_model_contracts(
    names: list[str] | None = None, *, forward: bool = False, backward: bool = False
) -> list[ContractFailure]:
    """Construct selected models and optionally run their minimal forward pass."""
    failures: list[ContractFailure] = []
    torch.set_num_threads(1)
    for name in names or MODEL_CATALOG.names():
        stage = "load"
        try:
            spec = MODEL_CATALOG.get(name)
            task = _task_for(spec)
            cfg = SimpleNamespace(task=task)
            params = _params_for(spec)
            stage = "construct"
            execute_forward = forward or backward
            for seed in spec.contract_seeds if execute_forward else spec.contract_seeds[:1]:
                torch.manual_seed(seed)
                with contextlib.redirect_stdout(io.StringIO()):
                    model = spec.build(cfg, params)
                if execute_forward:
                    stage = f"{'backward' if backward else 'forward'}(seed={seed})"
                    with contextlib.redirect_stdout(io.StringIO()):
                        _forward_contract(
                            model,
                            spec,
                            task,
                            spec.validate_params(params),
                            backward=backward,
                        )
                del model
        except Exception as exc:
            failures.append(ContractFailure(name, stage, f"{type(exc).__name__}: {exc}"))
        finally:
            gc.collect()
    return failures
