"""Visualize model forecasts vs. ground truth for a few test cases.

Given a run config, this builds the test loader and model exactly like the
runner, loads the best checkpoint trained for that (dataset, model) if present,
runs inference on a handful of test windows, and plots — per sample — the input
history, the true future, and the predicted future for one channel. Outputs a
PNG to ``work_dirs/<dataset>/<model>/cases.png`` (or ``--out``).

This is the "one-click case visualization" companion to the sweep-based
ablation / hyperparameter runs (see ``docs/en/experiments.md``).

Examples
--------
    # After training, e.g.  uv run modern-tsf --config configs/runs/run_single_data.toml
    uv run python tool/visualize_predictions.py \
        --config configs/runs/run_single_data.toml --num-samples 4 --channel -1

    # Point at a specific checkpoint:
    uv run python tool/visualize_predictions.py --config <cfg> \
        --checkpoint work_dirs/ETTh1/DLinear/checkpoints/<run_id>/best_checkpoint.pth
"""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from benchmark.config import load_config
from benchmark.registry.loader import register_from_config
from benchmark.registry.models import MODEL_REGISTRY
from benchmark.runner.trainer import _call_model
from data.provider import build_data_loader


def _find_checkpoint(work_dir: str, dataset: str, model: str) -> str | None:
    """Return the most recent ``best_checkpoint.pth`` for a (dataset, model)."""
    pattern = os.path.join(work_dir, dataset, model, "checkpoints", "*", "best_checkpoint.pth")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime)
    return matches[-1] if matches else None


def main() -> None:
    """Plot forecast vs. ground truth for a few test windows."""
    p = argparse.ArgumentParser(description="Visualize forecasts vs ground truth")
    p.add_argument("--config", required=True, help="Run config TOML")
    p.add_argument("--num-samples", type=int, default=4, help="Number of test windows to plot")
    p.add_argument("--channel", type=int, default=-1, help="Channel/node index to plot (-1 = last)")
    p.add_argument("--checkpoint", default=None, help="Checkpoint path (default: auto-find in work_dirs)")
    p.add_argument("--out", default=None, help="Output PNG path")
    args = p.parse_args()

    loaded = load_config(args.config)[0]
    config = loaded.config
    register_from_config(config)

    dataset_name = config.dataset.name
    model_name = config.model.name
    size = (config.task.seq_len, config.task.label_len, config.task.pred_len)
    dataset_params = (
        config.dataset.params.model_dump()
        if hasattr(config.dataset.params, "model_dump")
        else dict(config.dataset.params)
    )
    test_set, test_loader = build_data_loader(
        dataset_name, config.dataset.root_path, config.dataset.data_path, size,
        "test", config.task.features, dataset_params, args.num_samples, 0,
    )

    model_factory, params_schema = MODEL_REGISTRY.get(model_name)
    params = config.model.params
    if params_schema is not None:
        params = params_schema.model_validate(params).model_dump()
    if getattr(test_set, "adj_mx", None) is not None:
        params["adj_mx"] = test_set.adj_mx
    if getattr(test_set, "num_nodes", None) is not None:
        params.setdefault("num_nodes", test_set.num_nodes)
    model = model_factory(config, params)

    ckpt = args.checkpoint or _find_checkpoint(
        config.experiment.work_dir, dataset_name, model_name
    )
    if ckpt and os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        print(f"Loaded checkpoint: {ckpt}")
    else:
        print("WARNING: no checkpoint found — plotting with UNTRAINED weights. "
              "Train the model first (uv run modern-tsf --config ...).")
    model.eval()

    # One batch is enough — batch_size was set to num_samples above.
    batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(test_loader))
    batch_x = batch_x.float()
    batch_y = batch_y.float()
    batch_x_mark = batch_x_mark.float()
    batch_y_mark = batch_y_mark.float()
    pred_len = config.task.pred_len
    label_len = config.task.label_len
    dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :])
    dec_inp = torch.cat([batch_y[:, :label_len, :], dec_inp], dim=1)
    with torch.no_grad():
        outputs = _call_model(model, batch_x, batch_x_mark, dec_inp, batch_y_mark)
    outputs = outputs[:, -pred_len:, :].cpu().numpy()
    trues = batch_y[:, -pred_len:, :].cpu().numpy()
    hist = batch_x.cpu().numpy()

    ch = args.channel
    n = min(args.num_samples, outputs.shape[0])
    seq_len = hist.shape[1]
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.4 * n), squeeze=False)
    for i in range(n):
        ax = axes[i][0]
        h = hist[i, :, ch]
        t = trues[i, :, ch]
        pr = outputs[i, :, ch]
        ax.plot(range(seq_len), h, color="gray", label="history")
        ax.plot(range(seq_len, seq_len + pred_len), t, color="C0", label="ground truth")
        ax.plot(range(seq_len, seq_len + pred_len), pr, color="C1", linestyle="--", label="forecast")
        ax.axvline(seq_len - 0.5, color="k", linewidth=0.6, alpha=0.4)
        ax.set_title(f"{model_name} · {dataset_name} · case {i} · channel {ch}", fontsize=9)
        if i == 0:
            ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()

    out = args.out or os.path.join(
        config.experiment.work_dir, dataset_name, model_name, "cases.png"
    )
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"Wrote {out}  ({n} cases, channel {ch})")


if __name__ == "__main__":
    main()
