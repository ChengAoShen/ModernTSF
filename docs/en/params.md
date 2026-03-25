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

### [training.checkpoint]

- `strategy` (str): checkpoint strategy, e.g. `"best"`.
- `save_k` (int): number of checkpoints to keep.

## [dataset]

- `name` (str): dataset name registered in `DATASET_NAME_MAP`.
- `root_path` (str): dataset root directory.
- `data_path` (str): dataset file name (empty string for pre-split datasets).
- `params` (dict): dataset-specific parameters validated by the dataset schema.

### Common dataset params

Most datasets accept:

- `target` (str): target column name or index.
- `scale` (bool): whether to scale features.
- `split_ratio` (list[float]): train/val/test split ratios.

### Dataset-specific params

`periodic` (synthetic — create `configs/datasets/periodic.toml` with the params below)

- `channel_number` (int): number of channels.
- `num_samples` (int): number of independent samples to generate.
- `period` (int): period in timesteps.
- `noise_std` (float): Gaussian noise standard deviation.
- `amplitude_range` (list[float]): min/max amplitude range.
- `phase_range` (list[float]): phase range in radians (e.g. `[0, 2*pi]`).
- `cycle_start_mode` (str): start mode, `"random"` or fixed.
- `random_phase` (bool): whether to randomize phase per sample.

`ETT` (`configs/datasets/etth1.toml`, `etth2.toml`, `ettm1.toml`, `ettm2.toml`)

- Uses the common params above. ETT data is loaded from CSV and split in the original paper ratio.

`traffic` / `weather` / `electricity`

- Uses the common params above. CSV must include a `date` column for time features.

`solar`

- Uses the common params above. The solar dataset is loaded from a text file.

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

## [model]

- `name` (str): model name registered in `MODEL_NAME_MAP`.
- `params` (dict): model-specific parameters validated by the model schema.

The exact parameters are defined under `src/models/<model>/schema.py` and used in the model registry.

## [evaluation]

- `metrics` (list[str]): metric names resolved from `METRIC_NAME_MAP`.
- `enable_profile` (bool): whether to enable profiling.

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
