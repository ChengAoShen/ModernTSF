# Parameters reference

This document explains the meaning of each TOML section and field. The source of truth for defaults is `configs/base.toml` and the Pydantic schemas under `src/benchmark/config/schema/`.

## [experiment]

- `description` (str): free-form text for the run description.
- `random_seed` (int): global seed used for reproducibility.
- `work_dir` (str): base output directory for checkpoints, CSV summaries, and profiles.

### [experiment.runtime]

- `device` (str): runtime device, usually `"cuda"` or `"cpu"`.
- `use_multi_gpu` (bool): whether to enable multi-GPU (if supported by model).
- `device_ids` / `gpus` (list[int]): GPU ids. The config accepts `gpus` as an alias.
- `amp` (bool): enable automatic mixed precision.
- `num_workers` (int): DataLoader workers.

## [task]

- `seq_len` (int): input sequence length.
- `label_len` (int): decoder warm-up length (used by some models).
- `pred_len` (int): prediction horizon length.
- `features` (str): `"M"`, `"S"`, or `"MS"`.
  - `M`: multivariate input and output.
  - `S`: single target variable.
  - `MS`: multivariate input, single target output.
- `inverse` (bool): whether to inverse-transform outputs for evaluation if supported.

## [training]

- `epochs` (int): number of epochs to train.
- `batch_size` (int): batch size.
- `loss` (str): loss name resolved from `LOSS_NAME_MAP`.
- `loss_params` (dict): keyword args passed to the loss constructor (e.g. `reduction`).
- `patience` (int): early stopping patience (epochs).

### [training.optimizer]

- `name` (str): optimizer name (e.g. `Adam`).
- `lr` (float): learning rate.
- `weight_decay` (float): weight decay.
- `lradj` (str): learning-rate schedule name (if used).
- `params` (dict): extra optimizer keyword args.

### [training.tricks] (optional — all disabled by default; omitting the section changes nothing)

Pluggable training callbacks (`src/benchmark/runner/callbacks.py`):

- `grad_clip_norm` (float): clip gradient norm before the optimizer step.
- `grad_clip_norm_type` (float): norm type for clipping (default 2).
- `grad_accum_steps` (int): accumulate gradients over N micro-batches (larger effective batch).
- `[training.tricks.curriculum]`: `enabled`, `warmup_epochs`, `step_size`, `cl_epochs` — progressively grow the supervised horizon (BasicTS scheme), capped at `pred_len`.

**Auxiliary loss:** if a model exposes `self.aux_loss` (or `last_moe_loss` / `last_aux_loss`) as a finite scalar tensor, the trainer adds it to the training loss automatically (a no-op for models that don't). Useful for MoE-balance / KL / regularization terms (e.g. Pathformer, TimeFilter).

The performance summary also records `fit_time` and `inference_time`.

### [training.checkpoint]

- `strategy` (str): checkpoint strategy, e.g. `"best"`.
- `save_k` (int): number of checkpoints to keep.

## [dataset]

- `name` (str): dataset name registered in `DATASET_NAME_MAP`.
- `alias` (str | None): optional display alias used in CSV summaries and log output (e.g. `"gift_eval/bizitobs_application"`). Defaults to `null` (the `name` value is used).
- `root_path` (str): dataset root directory. Default: `"./data/"`.
- `data_path` (str): dataset file name relative to `root_path`. Set to `""` for pre-split and pre-processed datasets.
- `params` (dict): dataset-specific parameters validated by the dataset schema.

### Common dataset params

Most single-file datasets accept:

- `target` (str): target column name or index.
- `scale` (bool): whether to apply `StandardScaler`. Default: `true`.
- `split_ratio` (list[float]): train/val/test split ratios (proportional or absolute). Default varies by dataset.
- `norm_each_channel` (bool, default `false`): compute per-channel mean/std on the train split instead of the shared scaler (opt-in; default off = unchanged).
- `target_channel` (int|null, default `null`): anchor normalization/inverse-transform on this channel so the target and covariate channels keep separate statistics (useful for the covariate task mode).

### Dataset-specific params

`ETT` (`configs/datasets/etth1.toml`, `etth2.toml`, `ettm1.toml`, `ettm2.toml`)

- Uses the common params above.
- Default `split_ratio`: `[12.0, 4.0, 4.0]` (months, matching the original paper split).

`traffic` / `weather` / `electricity`

- Uses the common params above. CSV must include a `date` column for time features.
- Default `split_ratio`: `[0.7, 0.1, 0.2]`.

`solar`

- Uses the common params above. The solar dataset is loaded from a text file, not a CSV.
- Default `split_ratio`: `[0.7, 0.1, 0.2]`.

`periodic` (synthetic — create `configs/datasets/periodic.toml` with the params below)

- `target` (str): unused for generation but required by the schema. Default: `"OT"`.
- `scale` (bool): whether to scale. Default: `true`.
- `split_ratio` (list[float]): split ratios. Default: `[0.7, 0.1, 0.2]`.
- `channel_number` (int): number of channels. Default: `1`.
- `num_samples` (int): number of independent samples to generate. Default: `1024`.
- `period` (int): period in timesteps. Default: `24`.
- `noise_std` (float): Gaussian noise standard deviation. Default: `0.1`.
- `amplitude_range` (list[float]): min/max amplitude range. Default: `[0.5, 1.5]`.
- `phase_range` (list[float]): phase range in radians. Default: `[0.0, 6.283...]` (0 to 2π).
- `cycle_start_mode` (str): start mode, `"random"` or fixed. Default: `"random"`.
- `random_phase` (bool): whether to randomize phase per sample. Default: `true`.

`trend` (synthetic — create `configs/datasets/trend.toml` with the params below)

- `target` (str): unused for generation. Default: `"OT"`.
- `scale` (bool): Default: `true`.
- `split_ratio` (list[float]): Default: `[0.7, 0.1, 0.2]`.
- `channel_number` (int): number of channels. Default: `1`.
- `num_samples` (int): number of independent samples. Default: `1024`.
- `degree_min` (int): minimum polynomial degree. Default: `2`.
- `degree_max` (int): maximum polynomial degree. Default: `6`.
- `coeff_range` (list[float]): coefficient sampling range. Default: `[-0.8, 0.8]`.
- `noise_std` (float): Gaussian noise standard deviation. Default: `0.1`.
- `normalize_t` (bool): whether to normalize the time axis to `[0, 1]`. Default: `true`.

`presplit`

- `target` (str): target column name.
- `scale` (bool): whether to scale features (scaler is always fitted on `train.csv`).
- No `split_ratio` — the folder must contain `train.csv`, `val.csv`, and `test.csv`.
- `root_path` must point to the folder containing the three files. Set `data_path = ""`.

Example config:

```toml
[dataset]
name = "presplit"
root_path = "./dataset/my_dataset"
data_path = ""

[dataset.params]
target = "OT"
scale = true
```

`pre_processed`

- No `[dataset.params]` fields — all windowing is handled by `tool/pre_process.py`.
- `root_path` must point to the directory containing the `.npz` files.
- Set `data_path = ""`.

Example config:

```toml
[dataset]
name = "pre_processed"
root_path = "./dataset/my_dataset_npy"
data_path = ""
```

`gift_eval`

- `scale` (bool): whether to apply `StandardScaler` (fitted on training data). Default: `true`.
- `windows` (int | None): number of rolling test windows. `null` means auto-calculate following the GIFT-EVAL protocol. Default: `null`.
- Use `alias` at the `[dataset]` level to set a human-readable name in CSV summaries.

Example config:

```toml
[dataset]
name = "gift_eval"
alias = "gift_eval/bizitobs_application"
root_path = "./dataset/gift_eval"
data_path = "bizitobs_application"

[dataset.params]
scale = true
```

## [model]

- `name` (str): model name registered in `MODEL_NAME_MAP`.
- `params` (dict): model-specific parameters validated by the model schema.

The exact parameters are defined under `src/models/<model>/schema.py` and used in the model registry.

## [evaluation]

- `metrics` (list[str]): metric names resolved from `METRIC_NAME_MAP`. All metrics are always computed; this list selects which ones are written to `performance.csv`. The Pydantic schema default is `["mae", "mse", "rmse", "mape", "mspe"]`, but `configs/base.toml` widens it to `["mae", "mse", "rmse", "mape", "mspe", "corr", "rse", "wape", "smape"]` (so every run extending `base.toml` records all nine). Add `"mase"` explicitly to also record it.
- `enable_profile` (bool): whether to run the model profiler after evaluation. Default: `false`.
- `strategy` (`"fixed"` | `"rolling"`): evaluation strategy. Default `"fixed"` — the historical fixed-window evaluation that iterates the test `DataLoader` once. `"rolling"` opts into a TFB-style rolling forecast over the test split (see below). Default behavior is unchanged when omitted.

### Rolling forecast (`strategy = "rolling"`)

When `strategy = "rolling"`, the evaluator walks the test split: it consumes a `seq_len` input window, predicts `pred_len` steps, advances the window by `stride`, and repeats for up to `num_rollings` rollings (or until the test data is exhausted). The same metrics are then computed via `collect_metrics`. The rolling sub-config lives under `[evaluation.rolling]`:

- `horizon` (int | null): number of predicted steps scored per rolling. `null` (default) uses the full `pred_len`; clamped to `pred_len` (the model only emits `pred_len` steps per call).
- `stride` (int): steps the input window advances between rollings. Default `1`.
- `num_rollings` (int | null): maximum number of rollings. `null` (default) rolls until the test data runs out.

Rolling runs add an `eval_strategy=rolling` column to `performance.csv`; fixed runs leave the CSV header unchanged.

```toml
[evaluation]
strategy = "rolling"

[evaluation.rolling]
horizon = 96
stride = 1
num_rollings = 100
```

### Available metrics

| Name | Description |
|---|---|
| `mae` | Mean absolute error |
| `mse` | Mean squared error |
| `rmse` | Root mean squared error |
| `mape` | Mean absolute percentage error |
| `mspe` | Mean squared percentage error |
| `corr` | Mean per-channel Pearson correlation |
| `rse` | Relative squared error |
| `wape` | Weighted absolute percentage error (scale-free) |
| `smape` | Symmetric mean absolute percentage error |
| `mase` | Mean absolute scaled error (opt-in; eval-window naive baseline — see note) |

> `mase` is available but not in the default list: at evaluation time only the
> prediction window is available, so the naive baseline is built from the test
> targets (not in-sample history) — values aren't comparable to the textbook
> in-sample-scaled MASE. Add it to `[evaluation] metrics` explicitly if wanted.

### Available losses

Resolved from `LOSS_NAME_MAP` via `loss`. Standard: `mae`, `mse` (+ `freq_mae`,
`freq_weighted_mae`). **Masked** variants `masked_mae`, `masked_mse`,
`masked_rmse` ignore positions flagged by an optional `targets_mask` (1=valid,
0=ignore), normalizing the mask by its mean (BasicTS convention) so the loss is
unbiased w.r.t. the valid-entry count; with no mask they equal their plain
counterparts. Useful for traffic/missing-value forecasting.

### Adjacency normalization (graph models)

Node-structured datasets inject a raw `adj_mx` into the model factory. Set
`[dataset.params] adj_norm = "<scheme>"` to normalize it first, where `<scheme>`
∈ `sym_norm_lap` | `scaled_laplacian` | `gcn` | `transition` | `reverse_transition`
(see `src/models/_external/adj_norm.py`). Default (unset) injects the raw matrix.

### Profiling (`enable_profile = true`)

When enabled, the profiler runs on the test `DataLoader` after evaluation and writes two outputs per run:

- `work_dirs/<dataset>/<model>/profiles/<run_id>.txt` — human-readable report with three sections: architecture/parameter summary (via `torchinfo`), FLOPs (via `fvcore`, MACs in millions), and CUDA latency/throughput (50 timed forward passes after 10 warmup passes, skipped on CPU).
- `work_dirs/<dataset>/<model>/profile.csv` — structured CSV row appended per run.

Columns written to `profile.csv`:

| Column | Description |
|---|---|
| `total_params` | Total parameter count |
| `trainable_params` | Trainable parameter count |
| `non_trainable_params` | Non-trainable parameter count |
| `total_mult_adds_mb` | Total multiply-adds (MB, from torchinfo) |
| `total_macs_m` | Total MACs in millions (from fvcore) |
| `dynamic_vram_mb` | Dynamic VRAM allocated during forward pass (MB) |
| `peak_vram_mb` | Total peak VRAM allocated (MB) |
| `reserved_vram_mb` | Total reserved VRAM (MB) |
| `latency_avg_ms` | Average forward-pass latency (ms) |
| `throughput_samples_sec` | Throughput in samples/sec |

`torchinfo` and `fvcore` are optional. If not installed, the corresponding fields are omitted from the report (no error is raised). CUDA latency columns are omitted when running on CPU.

## [sweep]

Defines a sweep over configuration keys. Supported formats:

```toml
[sweep]
experiment.random_seed = [0, 1, 2]
```

```toml
[sweep.task]
pred_len = [96, 192, 336, 720]
```

Sweeps expand into the cartesian product of values, and each expanded config is validated and run.
