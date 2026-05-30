# Workflow scripts

Shell scripts for common multi-step workflows live in `scripts/`. Both scripts keep `set -euo pipefail` and detect the repo root automatically via `ROOT_DIR`.

---

## `run_multi_configs.sh`

Run one or more ModernTSF experiment configs sequentially on a chosen GPU.

### Usage

```bash
[GPU_IDS=<ids>] bash scripts/run_multi_configs.sh [config ...]
```

### Arguments

| Argument | Description | Default |
|---|---|---|
| `config ...` | One or more TOML config paths (relative to repo root). | `configs/runs/run_single_data.toml` |

### Environment variables

| Variable | Description | Default |
|---|---|---|
| `GPU_IDS` | Value forwarded to `CUDA_VISIBLE_DEVICES`. | `0` |

### Examples

```bash
# Run the default config on GPU 0
bash scripts/run_multi_configs.sh

# Run two sweep configs on GPU 1
GPU_IDS=1 bash scripts/run_multi_configs.sh configs/runs/sweep_data.toml configs/runs/sweep_model.toml
```

### Notes

- `-h` / `--help` prints the embedded usage header and exits.
- Each config is executed in sequence; the script aborts on the first failure (`set -e`).
- `cd "${ROOT_DIR}"` is performed internally, so paths are always resolved relative to the repo root regardless of where the script is invoked.

---

## `aggregate_and_plot.sh`

Aggregate per-run CSVs for a dataset / prediction-length combination and render a bubble chart in one step. Internally calls `tool/aggregate_results.py` then `tool/plot_bubble.py`.

### Usage

```bash
[DATASET=… PRED_LEN=… X=… Y=… SIZE=… OUT_CSV=… OUT_SVG=…] bash scripts/aggregate_and_plot.sh [DATASET] [PRED_LEN]
```

Positional arguments take precedence over environment variables of the same name.

### Arguments

| Argument | Description | Default |
|---|---|---|
| `DATASET` | Dataset name key (positional or `$DATASET` env). | `ETTh1` |
| `PRED_LEN` | Prediction length (positional or `$PRED_LEN` env). | `96` |

### Environment variables

| Variable | Description | Default |
|---|---|---|
| `X` | Bubble x-axis field. | `latency_avg_ms` |
| `Y` | Bubble y-axis field. | `mse` |
| `SIZE` | Bubble size field. | `total_params` |
| `OUT_CSV` | Aggregated CSV output path. | `work_dirs/${DATASET}/results_all.csv` |
| `OUT_SVG` | Bubble chart output path. | `work_dirs/plots/bubble_${DATASET}_pl${PRED_LEN}.svg` |

### Examples

```bash
# Default: ETTh1, pred_len=96
bash scripts/aggregate_and_plot.sh

# Positional args
bash scripts/aggregate_and_plot.sh ETTh1 96

# Env-var overrides
DATASET=weather PRED_LEN=720 Y=mae bash scripts/aggregate_and_plot.sh
```

### Notes

- `-h` / `--help` prints the embedded usage header and exits.
- The script `cd`s to `ROOT_DIR` before running any tools, so relative paths in `OUT_CSV` / `OUT_SVG` are always anchored to the repo root.
- Bubble axes default to log scale for `X`, `Y`, and `SIZE`; bubbles are colored and labeled by `model`.
- Aggregate step filters on `pred_len=${PRED_LEN}` and retains fields `model,seq_len,pred_len,mse,mae` (perf) and `latency_avg_ms,throughput_samples_sec,total_params,peak_vram_mb` (profile).

---

## `detect_hardware.sh`

Detect the GPU / driver / CUDA version and recommend a uv PyTorch backend tag for `UV_TORCH_BACKEND` (`cpu | cu118 | cu121 | cu124 | cu126 | cu128`). Used by the `setup-env` skill; see [setup-env.md](setup-env.md).

### Usage

```bash
bash scripts/detect_hardware.sh             # human-readable report
bash scripts/detect_hardware.sh --backend   # print only the backend tag
UV_TORCH_BACKEND="$(bash scripts/detect_hardware.sh --backend)" uv sync --python 3.12
```

### Output

```
gpu=NVIDIA GeForce RTX 4090
driver=550.54.15
cuda=12.4
backend=cu124
```

### Notes

- No GPU / no `nvidia-smi` on `PATH` → reports `backend=cpu`.
- Maps the driver's max CUDA version to the highest available wheel backend ≤ that version (CUDA is backward compatible within a major release).
- Read-only: it never installs anything — it only reports and recommends.
