"""Single-run orchestration: data, model, training, evaluation."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import torch

from benchmark.registry import MODEL_REGISTRY
from benchmark.evaluation import profile_model
from benchmark.evaluation.profile import parse_profile_report_file
from benchmark.utils import default_summary_row, set_seed, write_csv_summary
from benchmark.utils.results import _flatten_params
from data.provider import build_data_loader
from benchmark.runner.trainer import train
from benchmark.runner.evaluator import evaluate


@dataclass
class RunResult:
    """Aggregate results from a single training/evaluation run.

    Parameters
    ----------
    metrics : dict[str, float]
        Evaluation metrics computed on the test split.
    train_time_sec : float
        Training wall-clock time in seconds.
    test_time_sec : float
        Evaluation wall-clock time in seconds.
    checkpoint_path : str
        Path to the best checkpoint on disk.
    run_id : str
        Unique identifier for the run.
    """

    metrics: dict[str, float]
    train_time_sec: float
    test_time_sec: float
    checkpoint_path: str
    run_id: str


def _build_device(runtime) -> torch.device:
    """Resolve the compute device from runtime settings.

    Parameters
    ----------
    runtime : ExperimentRuntimeConfig
        Runtime config with device selection options.

    Returns
    -------
    torch.device
        Resolved device.
    """
    if runtime.device == "cuda" and torch.cuda.is_available():
        if runtime.use_multi_gpu:
            return torch.device("cuda")
        return torch.device(f"cuda:{runtime.device_ids[0]}")
    if (
        runtime.device == "mps"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_data_path(root_path: str, data_path: str) -> str:
    """Resolve dataset path to an absolute file path.

    Parameters
    ----------
    root_path : str
        Root directory for datasets.
    data_path : str
        File name or absolute path.

    Returns
    -------
    str
        Absolute path to the dataset file.
    """
    if os.path.isabs(data_path):
        return data_path
    return os.path.join(root_path, data_path)


def run_one(
    config,
    raw: dict,
    sweep_keys: list[str] | None = None,
    config_name: str | None = None,
) -> RunResult:
    """Execute a full training/evaluation run for one config.

    Parameters
    ----------
    config : RootConfig
        Validated configuration object.
    raw : dict
        Raw expanded config dictionary (used for sweep columns).
    sweep_keys : list[str] | None, optional
        Dot-delimited keys from the sweep section.
    config_name : str | None, optional
        Deprecated output grouping hint (ignored).

    Returns
    -------
    RunResult
        Metrics and artifact paths for the run.
    """
    dataset_name = config.dataset.alias or config.dataset.name
    dataset_registry_name = config.dataset.name

    if raw and sweep_keys:
        flattened = _flatten_params(raw)
        sweep_parts = [
            f"{key}={flattened[key]}" for key in sweep_keys if key in flattened
        ]
    else:
        sweep_parts = []

    summary_parts = [
        f"model={config.model.name}",
        f"dataset={dataset_name}",
        f"seq_len={config.task.seq_len}",
        f"pred_len={config.task.pred_len}",
        f"seed={config.experiment.random_seed}",
    ]
    if sweep_parts:
        summary_parts.append(f"sweep: {', '.join(sweep_parts)}")
    print(f"Run config | {' | '.join(summary_parts)}")

    set_seed(config.experiment.random_seed)
    device = _build_device(config.experiment.runtime)
    print(f"Using device: {device}")

    data_path = _resolve_data_path(config.dataset.root_path, config.dataset.data_path)
    root_path = os.path.dirname(data_path)
    data_file = os.path.basename(data_path)

    size = (config.task.seq_len, config.task.label_len, config.task.pred_len)
    if hasattr(config.dataset.params, "model_dump"):
        dataset_params = config.dataset.params.model_dump()
    else:
        dataset_params = dict(config.dataset.params)

    train_set, train_loader = build_data_loader(
        dataset_registry_name,
        root_path,
        data_file,
        size,
        "train",
        config.task.features,
        dataset_params,
        config.training.batch_size,
        config.experiment.runtime.num_workers,
    )
    vali_set, vali_loader = build_data_loader(
        dataset_registry_name,
        root_path,
        data_file,
        size,
        "val",
        config.task.features,
        dataset_params,
        config.training.batch_size,
        config.experiment.runtime.num_workers,
    )
    test_set, test_loader = build_data_loader(
        dataset_registry_name,
        root_path,
        data_file,
        size,
        "test",
        config.task.features,
        dataset_params,
        config.training.batch_size,
        config.experiment.runtime.num_workers,
    )

    model_factory, params_schema = MODEL_REGISTRY.get(config.model.name)
    params = config.model.params
    if params_schema is not None:
        params = params_schema.model_validate(params).model_dump()

    model = model_factory(config, params).to(device)
    if config.experiment.runtime.use_multi_gpu and device.type == "cuda":
        model = torch.nn.DataParallel(
            model, device_ids=config.experiment.runtime.device_ids
        )
    optimizer_cls = getattr(torch.optim, config.training.optimizer.name)
    optimizer_kwargs = {
        "lr": config.training.optimizer.lr,
        "weight_decay": config.training.optimizer.weight_decay,
    }
    optimizer_kwargs.update(config.training.optimizer.params)
    optimizer = optimizer_cls(model.parameters(), **optimizer_kwargs)

    run_id = (
        f"{config.model.name}_{dataset_name}_sl{config.task.seq_len}_"
        f"pl{config.task.pred_len}_seed{config.experiment.random_seed}_{int(time.time())}"
    )
    output_group = os.path.join(dataset_name, config.model.name)
    model_dir = os.path.join(config.experiment.work_dir, output_group)
    checkpoint_dir = os.path.join(model_dir, "checkpoints", run_id)
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_result = train(
        model=model,
        train_loader=train_loader,
        vali_loader=vali_loader,
        device=device,
        epochs=config.training.epochs,
        patience=config.training.patience,
        loss_name=config.training.loss,
        loss_params=config.training.loss_params,
        optimizer=optimizer,
        lradj=config.training.optimizer.lradj,
        base_lr=config.training.optimizer.lr,
        total_epochs=config.training.epochs,
        label_len=config.task.label_len,
        pred_len=config.task.pred_len,
        features=config.task.features,
        use_amp=config.experiment.runtime.amp,
        checkpoint_dir=checkpoint_dir,
        checkpoint_cfg=config.training.checkpoint,
    )

    metrics, test_time = evaluate(
        model=model,
        data_loader=test_loader,
        device=device,
        label_len=config.task.label_len,
        pred_len=config.task.pred_len,
        features=config.task.features,
        inverse=config.task.inverse,
        dataset=test_set,
    )

    if config.evaluation.metrics:
        metrics = {k: v for k, v in metrics.items() if k in config.evaluation.metrics}

    metrics_str = ", ".join(f"{k}:{v:.4f}" for k, v in metrics.items())
    print(f"Test metrics | {metrics_str}")

    summary_path = os.path.join(model_dir, "performance.csv")
    print(f"Writing CSV summary to: {summary_path}")
    summary_row = default_summary_row(
        {
            "run_id": run_id,
            "dataset": dataset_name,
            "model": config.model.name,
            "seq_len": config.task.seq_len,
            "pred_len": config.task.pred_len,
            "seed": config.experiment.random_seed,
            "train_time_sec": train_result.train_time_sec,
            "test_time_sec": test_time,
        },
        metrics,
        raw=raw,
        sweep_keys=sweep_keys,
    )
    write_csv_summary(summary_path, summary_row)

    if config.evaluation.enable_profile:
        os.makedirs(os.path.join(model_dir, "profiles"), exist_ok=True)
        profile_path = os.path.join(model_dir, "profiles", f"{run_id}.txt")
        profile_model(
            model=model,
            data_loader=test_loader,
            device=device,
            label_len=config.task.label_len,
            pred_len=config.task.pred_len,
            save_path=profile_path,
        )
        profile_metrics = parse_profile_report_file(profile_path)
        profile_row = {
            "run_id": run_id,
            "model": config.model.name,
            "dataset": dataset_name,
            "seq_len": config.task.seq_len,
            "pred_len": config.task.pred_len,
            "seed": config.experiment.random_seed,
            "train_time_sec": train_result.train_time_sec,
            "test_time_sec": test_time,
        }
        profile_row.update(profile_metrics)
        profile_header = [
            "run_id",
            "model",
            "dataset",
            "seq_len",
            "pred_len",
            "seed",
            "train_time_sec",
            "test_time_sec",
            "total_params",
            "trainable_params",
            "non_trainable_params",
            "total_mult_adds_mb",
            "total_macs_m",
            "dynamic_vram_mb",
            "peak_vram_mb",
            "reserved_vram_mb",
            "latency_avg_ms",
            "throughput_samples_sec",
        ]
        profile_csv_path = os.path.join(model_dir, "profile.csv")
        write_csv_summary(profile_csv_path, profile_row, header=profile_header)

    return RunResult(
        metrics=metrics,
        train_time_sec=train_result.train_time_sec,
        test_time_sec=test_time,
        checkpoint_path=train_result.best_model_path,
        run_id=run_id,
    )
