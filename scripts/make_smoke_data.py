"""Generate a tiny synthetic CSV for end-to-end smoke runs.

Writes ``dataset/smoke/smoke.csv`` with a ``date`` column (hourly) plus
``N`` numeric channels, the last named ``OT`` (the default target). Just
enough rows to form a handful of train/val/test windows.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


def main() -> None:
    """Create the synthetic smoke dataset."""
    rows = 400
    n_channels = 6
    rng = np.random.default_rng(0)

    dates = pd.date_range("2020-01-01", periods=rows, freq="h")
    t = np.arange(rows)
    data = {}
    for c in range(n_channels - 1):
        # Daily + weekly seasonality plus noise.
        series = (
            np.sin(2 * np.pi * t / 24 + c)
            + 0.5 * np.sin(2 * np.pi * t / (24 * 7))
            + 0.1 * rng.standard_normal(rows)
        )
        data[f"ch{c}"] = series
    data["OT"] = (
        np.cos(2 * np.pi * t / 24) + 0.1 * rng.standard_normal(rows)
    )

    df = pd.DataFrame({"date": dates, **data})
    out_dir = os.path.join("dataset", "smoke")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "smoke.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path}  shape={df.shape}")


def make_cauair_smoke() -> None:
    """Create a tiny synthetic CauAir-style node bundle for covariate-mode runs.

    Writes ``dataset/cauair_ccaq/{his.npz, adj_mx.npy, idx_{train,val,test}.npy}``.
    ``data`` is ``(T, N, C)`` with channel 0 the target value and channels
    ``1:`` per-node covariates — just enough windows to train one smoke epoch.
    No real air-quality data is downloaded; this only exists so the
    ``covariate`` task mode (CauAir / AirCade future-covariate path) and the
    node-structured ``cauair_st`` loader can be exercised end-to-end on CPU.
    """
    rng = np.random.default_rng(1)
    n_steps, n_nodes, n_channels = 400, 8, 8  # 1 value + 7 covariates
    t = np.arange(n_steps)
    arr = np.zeros((n_steps, n_nodes, n_channels), dtype=np.float32)
    for n in range(n_nodes):
        arr[:, n, 0] = (
            np.sin(2 * np.pi * t / 24 + n)
            + 0.5 * np.sin(2 * np.pi * t / (24 * 7))
            + 0.1 * rng.standard_normal(n_steps)
        )
        for c in range(1, n_channels):
            arr[:, n, c] = np.sin(
                2 * np.pi * t / (24 * (c + 1)) + n + c
            ) + 0.1 * rng.standard_normal(n_steps)
    flat = arr.reshape(-1, n_channels)
    mean, std = flat.mean(0), flat.std(0)
    adj = np.eye(n_nodes, dtype=np.float32)

    out_dir = os.path.join("dataset", "cauair_ccaq")
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "his.npz"), data=arr, mean=mean, std=std)
    np.save(os.path.join(out_dir, "adj_mx.npy"), adj)
    # Window centres valid for seq_len<=24 and pred_len<=24: [24, n_steps-25].
    centers = np.arange(24, n_steps - 25)
    n_tr, n_va = int(len(centers) * 0.7), int(len(centers) * 0.1)
    np.save(os.path.join(out_dir, "idx_train.npy"), centers[:n_tr])
    np.save(os.path.join(out_dir, "idx_val.npy"), centers[n_tr : n_tr + n_va])
    np.save(os.path.join(out_dir, "idx_test.npy"), centers[n_tr + n_va :])
    print(f"Wrote {out_dir}/his.npz  data={arr.shape}  windows={len(centers)}")


if __name__ == "__main__":
    main()
    make_cauair_smoke()
