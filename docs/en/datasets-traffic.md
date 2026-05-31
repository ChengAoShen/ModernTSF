# Traffic node + adjacency datasets (METR-LA, PEMS-BAY, PEMS0x)

These are spatiotemporal traffic graphs: a `(T, N)` value matrix (speed or flow
over `N` sensors) plus an `(N, N)` adjacency. ModernTSF loads them through the
existing **`cauair_st`** node loader — there is *no* dedicated traffic loader.
You convert the raw arrays into a node *bundle* with `tool/convert_traffic.py`,
then point one of the shipped configs at the output directory.

Configs (all use `name = "cauair_st"`, `input_dim = 3`, `scale = false`):

| Config | `root_path` |
|---|---|
| `configs/datasets/metr_la.toml`  | `./dataset/metr_la` |
| `configs/datasets/pems_bay.toml` | `./dataset/pems_bay` |
| `configs/datasets/pems03.toml`   | `./dataset/pems03` |
| `configs/datasets/pems04.toml`   | `./dataset/pems04` |
| `configs/datasets/pems07.toml`   | `./dataset/pems07` |
| `configs/datasets/pems08.toml`   | `./dataset/pems08` |

`input_dim = 3` keeps the value channel plus the two calendar covariates
(`time_in_day`, `day_in_week`) appended by `--add-time`. `scale = false` keeps
those covariates raw — `time_in_day`/`day_in_week` are already in `[0, 1)` and
should *not* be z-scored. (If you want the value channel z-scored, scale it
upstream or write a separate config that flips `scale`.)

## Where to get the raw data

The data is download-on-demand; ModernTSF ships no copies. Common sources:

- **METR-LA / PEMS-BAY** — the DCRNN release
  (<https://github.com/liyaguang/DCRNN>) ships `metr-la.h5` / `pems-bay.h5` and
  `adj_mx.pkl` / `adj_mx_bay.pkl`. The Google-Drive mirror linked from that repo
  is the usual download. Convert the `.h5` to a `(T, N)` matrix (e.g. via
  `pandas.read_hdf(...).values`) and save it as `.npz`.
- **PEMS03 / PEMS04 / PEMS07 / PEMS08** — the ASTGCN / STSGCN release
  (<https://github.com/Davidham3/ASTGCN>, <https://github.com/Davidham3/STSGCN>)
  ships each as `pems0x.npz` (key `data`, shape `(T, N, F)`; channel 0 is flow)
  plus a distance CSV you can build the adjacency from, or a prebuilt
  `adj_mx.npy`.

## Convert into a bundle

```bash
# METR-LA: (T, N) speed matrix + a .pkl adjacency (3-tuple)
uv run python tool/convert_traffic.py \
    --values dataset/metr_la/metr-la.npz --values-key data \
    --adj    dataset/metr_la/adj_mx.pkl \
    --output-dir dataset/metr_la \
    --add-time --freq-min 5 \
    --seq-len 96 --pred-len 96

# PEMS04: (T, N, F) flow tensor (channel 0 is taken) + .npy adjacency
uv run python tool/convert_traffic.py \
    --values dataset/pems04/pems04.npz --values-key data \
    --adj    dataset/pems04/adj_mx.npy \
    --output-dir dataset/pems04 \
    --add-time --freq-min 5 \
    --seq-len 96 --pred-len 96
```

The converter writes the bundle the `cauair_st` loader expects:

```
<output-dir>/his.npz        # data (T, N, C), mean (C,), std (C,)
<output-dir>/adj_mx.npy      # (N, N) adjacency (graph models)
<output-dir>/idx_train.npy   # window-centre indices
<output-dir>/idx_val.npy
<output-dir>/idx_test.npy
```

### Adjacency formats

`--adj` accepts `.npy`, `.npz` (use `--adj-key`), or `.pkl`/`.pickle`. METR-LA
and PEMS-BAY ship `adj_mx.pkl` as a 3-tuple `(sensor_ids, sensor_id_to_idx,
adj_mx)`; the converter loads the pickle and takes the **last** element as the
`(N, N)` matrix. A bare-`ndarray` pickle is used as-is. Python-2-pickled files
fall back to `latin1` decoding.

### Split-ratio convention

Window centres are split **chronologically** (no shuffle) via `--splits`,
default `0.7,0.1,0.2` (train / val / test), so the test window always covers the
most recent data. This matches ModernTSF's single-file convention
(`split_ratio = [0.7, 0.1, 0.2]`). The canonical DCRNN split for METR-LA /
PEMS-BAY is `0.7 / 0.1 / 0.2` as well, so the default reproduces it; pass
`--splits` explicitly if you need a different convention.

## Run

The CLI takes a single `--config`, so point a run config at the dataset via
`extends`. For example, a run config (`configs/runs/your_traffic_run.toml`) with:

```toml
extends = ["../base.toml", "../datasets/metr_la.toml", "../models/GWNet.toml"]

[task]
mode = "spatiotemporal"
```

then run it:

```bash
uv run modern-tsf --config configs/runs/your_traffic_run.toml
```
