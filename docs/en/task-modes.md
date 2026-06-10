# Task modes

All ModernTSF tasks are **forecasting**; `task.mode` selects the *data setting*
— how a batch is shaped and what the model receives. The default is
`time_series`, so existing configs are unaffected.

```toml
[task]
mode = "time_series"   # or "spatiotemporal" | "covariate"
seq_len = 96
label_len = 0
pred_len = 96
```

## `time_series` (default)

Classic multivariate time-series forecasting. A batch is a value tensor
`(B, T, C)`; every channel is both an input and a target. Calendar features are
the raw stamps `(B, T, 6)` = `[year, month, day, weekday, hour, minute]` built
from the dataset's `date` column (zero-filled when absent). This is the
historical ModernTSF behaviour — nothing changes for existing models or
datasets.

## `spatiotemporal`

Node-structured forecasting. Each of the `N` nodes carries a value plus `F`
per-node covariates, so the natural input is `(B, T, N, 1 + F)`. ModernTSF
carries this through the standard four-item dataset contract by putting the
**value** in the series slots `(B, T, N)` and the **covariates** in the stamp
slots `(B, T, N, F)`:

```
__getitem__ -> (value_hist (T,N), value_fut (T,N), cov_hist (T,N,F), cov_fut (T,N,F))
```

A spatiotemporal model reconstructs `(B, T, N, 1 + F)` from the value and the
covariate marks. The only target is the value channel of all `N` nodes; the
output is `(B, pred_len, N)`.

Two flavours of covariate exist:

- **Calendar covariates** (`F = 2`, `[time_in_day, day_in_week]` in `[0, 1)`).
  `BiST`, `MAGE`, and `STOP` consume these as embedding indices. The
  `synthetic_st` dataset produces this layout.
- **Arbitrary covariates** (any `F`). `CauAir` / `AirCade` project covariates
  with linear layers, so meteorology-style covariates work directly.

## `covariate`

Like `spatiotemporal`, but the model also receives the **future** (known)
covariate block — the covariates over the prediction horizon — via the future
stamp `(B, pred_len, N, F)`. This is the decoder-side covariate input used by
forecasters that know future exogenous variables but not the future target,
such as the air-quality models (`CauAir`, `AirCade`), which know future
meteorology but not future pollutant values. Set `cov_dim = F` on these models
so the future block is sized correctly.

## Model / mode compatibility

| Model | time_series | spatiotemporal | covariate |
|---|:---:|:---:|:---:|
| `MoFo`, `PHAT`, and the stock forecasters | ✓ | | |
| `BiST`, `MAGE`, `STOP` | ✓ (calendar marks) | ✓ (calendar covariates) | |
| `CauAir`, `AirCade` | ✓ | ✓ | ✓ (future covariates) |

The model adapters are polymorphic: a 3-D mark `(B, T, 6)` is treated as raw
calendar stamps (time_series), while a 4-D mark `(B, T, N, F)` is treated as
node-structured covariates (spatiotemporal / covariate). See
`src/models/_external/marks.py`.

## Datasets per mode

- `time_series` — any CSV dataset (ETT, weather, custom, …).
- `spatiotemporal` — `synthetic_st` (calendar covariates) or `cauair_st`
  (CauAir / CCAQ meteorology).
- `covariate` — `cauair_st` (provides the future covariate block).
- The same CauAir data is also available as a plain time-series dataset
  `cauair_ts`, where the `N` node values become the `C` channels.

Tiny end-to-end smoke runs live in `configs/runs/smoke_st_bist.toml`
(spatiotemporal) and `configs/runs/smoke_cov_cauair.toml` (covariate).

## Graph models & the adjacency matrix

Graph models (e.g. STGCN, DCRNN, GraphWaveNet) need an `(N, N)` adjacency. A
node-structured dataset exposes it as `self.adj_mx` (loaded from `adj_mx.npy` in
the bundle, or `None` when absent). The runner reads it from the **train**
dataset and injects it into the model factory, so a graph model's
`registry.py` can pick it up:

```python
lambda cfg, params: Model(
    seq_len=cfg.task.seq_len, pred_len=cfg.task.pred_len,
    num_nodes=params["num_nodes"],          # injected from the dataset
    adj_mx=params.get("adj_mx"),            # (N, N) np.ndarray or None
    ...
)
```

`params["adj_mx"]` and `params["num_nodes"]` are added **after** schema
validation (see `src/benchmark/runner/run_one.py`), so they need not be declared
in the model's TOML — they come from the data. Non-graph models simply ignore
them.

### Optional adjacency normalization (`adj_norm`)

Set `adj_norm` under `[dataset.params]` to normalize the data-derived adjacency
before it is injected. The raw matrix is passed through unchanged when `adj_norm`
is unset, so existing graph models that build their own normalization are
unaffected. Supported schemes (`src/models/_external/adj_norm.py`):

| `adj_norm` | Function |
|---|---|
| `sym_norm_lap` / `symmetric_normalized_laplacian` | Symmetric normalized Laplacian |
| `scaled_laplacian` | Scaled Laplacian (Chebyshev) |
| `gcn` / `gcn_norm` | GCN renormalization (`D^-½ (A+I) D^-½`) |
| `transition` / `transition_matrix` | Random-walk transition matrix |
| `reverse_transition` / `reverse_transition_matrix` | Reverse random-walk transition |

```toml
[dataset.params]
adj_norm = "gcn"
```

To use a real traffic dataset (METR-LA, PEMS-BAY, PEMS0x), convert its raw value
matrix + adjacency into the node bundle with `tool/convert_traffic.py`, then
point a `cauair_st` dataset config at the output directory:

```bash
uv run python tool/convert_traffic.py \
    --values dataset/metr_la/metr-la.npz --values-key data \
    --adj dataset/metr_la/adj_mx.npy \
    --output-dir dataset/metr_la --add-time --freq-min 5
```

## Scope: forecasting only

All three modes are forecasting. Other task types — **imputation**, **anomaly
detection**, **classification**, and **foundation-model pretraining** — are
intentionally out of scope: each would need its own dataset format, task
contract, and evaluation protocol. There is no `task_name` parameter or
non-forecast branch anywhere in the codebase; the multi-task branches that ship
with some upstream TSLib-style models were stripped during porting.
