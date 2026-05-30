"""Convert a traffic-style spatiotemporal dataset into a node bundle.

ModernTSF's node-structured loader (``cauair_st`` / ``cauair_ts``) reads a
bundle of:

    <out_dir>/his.npz        # data (T, N, C); optional mean (C,), std (C,)
    <out_dir>/adj_mx.npy     # (N, N) adjacency, optional (graph models)
    <out_dir>/idx_train.npy  # window-centre indices
    <out_dir>/idx_val.npy
    <out_dir>/idx_test.npy

This converter builds that bundle from common public traffic datasets
(METR-LA, PEMS-BAY, PEMS03/04/07/08), which ship the value tensor and an
adjacency separately. Channel 0 of ``data`` is the target speed/flow; you can
append calendar covariates (time-of-day, day-of-week) with ``--add-time``.

Examples
--------
    # From a raw (T, N) speed matrix + an (N, N) adjacency:
    uv run python tool/convert_traffic.py \
        --values dataset/metr_la/metr-la.npz --values-key data \
        --adj dataset/metr_la/adj_mx.npy \
        --output-dir dataset/metr_la --add-time --freq-min 5

Then point a dataset config at it::

    [dataset]
    name = "cauair_st"          # node-structured (graph / spatiotemporal)
    root_path = "dataset/metr_la"
    data_path = ""

    [dataset.params]
    input_dim = 3               # value + [time_in_day, day_in_week]
"""

from __future__ import annotations

import argparse
import os

import numpy as np


def _load_array(path: str, key: str | None) -> np.ndarray:
    """Load an array from a ``.npy`` or ``.npz`` file."""
    if path.endswith(".npz"):
        bundle = np.load(path, allow_pickle=True)
        if key is None:
            key = list(bundle.keys())[0]
        return np.asarray(bundle[key])
    return np.asarray(np.load(path, allow_pickle=True))


def _add_time_features(values: np.ndarray, freq_min: int) -> np.ndarray:
    """Append ``[time_in_day, day_in_week]`` covariates to a ``(T, N, 1)`` tensor."""
    t, n, _ = values.shape
    steps_per_day = (24 * 60) // freq_min
    step = np.arange(t)
    tod = (step % steps_per_day) / steps_per_day
    dow = ((step // steps_per_day) % 7) / 7.0
    tod = np.broadcast_to(tod[:, None, None], (t, n, 1))
    dow = np.broadcast_to(dow[:, None, None], (t, n, 1))
    return np.concatenate([values, tod, dow], axis=-1).astype(np.float32)


def main() -> None:
    """Convert raw traffic arrays into a ModernTSF node bundle."""
    p = argparse.ArgumentParser(description="Convert traffic data to a node bundle")
    p.add_argument("--values", required=True, help="Path to the value array (.npy/.npz)")
    p.add_argument("--values-key", default=None, help="Key inside an .npz value file")
    p.add_argument("--adj", default=None, help="Path to the (N, N) adjacency (.npy/.npz)")
    p.add_argument("--adj-key", default=None, help="Key inside an .npz adjacency file")
    p.add_argument("--output-dir", required=True, help="Bundle output directory")
    p.add_argument("--seq-len", type=int, default=96, help="History length")
    p.add_argument("--pred-len", type=int, default=96, help="Forecast horizon")
    p.add_argument("--add-time", action="store_true", help="Append calendar covariates")
    p.add_argument("--freq-min", type=int, default=5, help="Minutes per step (for --add-time)")
    p.add_argument(
        "--splits", default="0.7,0.1,0.2", help="train,val,test ratios (comma-separated)"
    )
    args = p.parse_args()

    values = _load_array(args.values, args.values_key).astype(np.float32)
    if values.ndim == 2:  # (T, N) -> (T, N, 1)
        values = values[..., None]
    if values.ndim != 3:
        raise ValueError(f"values must be (T, N) or (T, N, C); got {values.shape}")
    if args.add_time:
        values = _add_time_features(values[..., :1], args.freq_min)

    t, n, c = values.shape
    mean = values.reshape(-1, c).mean(0)
    std = values.reshape(-1, c).std(0)

    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(os.path.join(args.output_dir, "his.npz"), data=values, mean=mean, std=std)

    if args.adj:
        adj = _load_array(args.adj, args.adj_key).astype(np.float32)
        np.save(os.path.join(args.output_dir, "adj_mx.npy"), adj)

    # Window centres valid for the chosen history/horizon, split chronologically.
    lo, hi = args.seq_len - 1, t - args.pred_len
    centers = np.arange(lo, hi)
    r_tr, r_va, _ = (float(x) for x in args.splits.split(","))
    n_tr, n_va = int(len(centers) * r_tr), int(len(centers) * r_va)
    np.save(os.path.join(args.output_dir, "idx_train.npy"), centers[:n_tr])
    np.save(os.path.join(args.output_dir, "idx_val.npy"), centers[n_tr : n_tr + n_va])
    np.save(os.path.join(args.output_dir, "idx_test.npy"), centers[n_tr + n_va :])

    adj_note = f", adj {n}x{n}" if args.adj else ", no adjacency"
    print(f"Wrote bundle to {args.output_dir}  data={values.shape}{adj_note}  windows={len(centers)}")


if __name__ == "__main__":
    main()
