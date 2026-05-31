# 工作流脚本

常用多步骤工作流的 Shell 脚本位于 `scripts/` 目录下。三个脚本均启用 `set -euo pipefail`；其中 `run_multi_configs.sh` 与 `aggregate_and_plot.sh` 还会通过 `ROOT_DIR` 自动检测仓库根目录。

---

## `run_multi_configs.sh`

在指定 GPU 上顺序运行一个或多个 ModernTSF 实验配置。

### 用法

```bash
[GPU_IDS=<ids>] bash scripts/run_multi_configs.sh [config ...]
```

### 参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `config ...` | 一个或多个 TOML 配置路径（相对于仓库根目录）。 | `configs/runs/run_single_data.toml` |

### 环境变量

| 变量 | 说明 | 默认值 |
|---|---|---|
| `GPU_IDS` | 传递给 `CUDA_VISIBLE_DEVICES` 的值。 | `0` |

### 示例

```bash
# 在 GPU 0 上运行默认配置
bash scripts/run_multi_configs.sh

# 在 GPU 1 上顺序运行两个 sweep 配置
GPU_IDS=1 bash scripts/run_multi_configs.sh configs/runs/sweep_data.toml configs/runs/sweep_model.toml
```

### 备注

- `-h` / `--help` 打印内嵌的用法说明后退出。
- 各配置按顺序执行，遇到第一个失败时脚本终止（`set -e`）。
- 脚本内部执行 `cd "${ROOT_DIR}"`，无论从哪里调用，路径均以仓库根目录为基准。

---

## `aggregate_and_plot.sh`

一步完成指定数据集与预测长度的结果汇总，并渲染气泡图。内部依次调用 `tool/aggregate_results.py` 和 `tool/plot_bubble.py`。

### 用法

```bash
[DATASET=… PRED_LEN=… X=… Y=… SIZE=… OUT_CSV=… OUT_SVG=…] bash scripts/aggregate_and_plot.sh [DATASET] [PRED_LEN]
```

位置参数优先于同名环境变量。

### 参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `DATASET` | 数据集名称键（位置参数或 `$DATASET` 环境变量）。 | `ETTh1` |
| `PRED_LEN` | 预测长度（位置参数或 `$PRED_LEN` 环境变量）。 | `96` |

### 环境变量

| 变量 | 说明 | 默认值 |
|---|---|---|
| `X` | 气泡图 x 轴字段。 | `latency_avg_ms` |
| `Y` | 气泡图 y 轴字段。 | `mse` |
| `SIZE` | 气泡大小字段。 | `total_params` |
| `OUT_CSV` | 汇总 CSV 输出路径。 | `work_dirs/${DATASET}/results_all.csv` |
| `OUT_SVG` | 气泡图输出路径。 | `work_dirs/plots/bubble_${DATASET}_pl${PRED_LEN}.svg` |

### 示例

```bash
# 默认：ETTh1，pred_len=96
bash scripts/aggregate_and_plot.sh

# 使用位置参数
bash scripts/aggregate_and_plot.sh ETTh1 96

# 使用环境变量覆盖
DATASET=weather PRED_LEN=720 Y=mae bash scripts/aggregate_and_plot.sh
```

### 备注

- `-h` / `--help` 打印内嵌的用法说明后退出。
- 脚本执行前会 `cd` 到 `ROOT_DIR`，`OUT_CSV` / `OUT_SVG` 中的相对路径始终以仓库根目录为基准。
- 气泡图的 `X`、`Y` 和 `SIZE` 轴默认使用对数刻度；气泡按 `model` 着色并添加标注。
- 汇总步骤按 `pred_len=${PRED_LEN}` 过滤，保留字段 `model,seq_len,pred_len,mse,mae`（性能）和 `latency_avg_ms,throughput_samples_sec,total_params,peak_vram_mb`（Profile）。

---

## `detect_hardware.sh`

检测 GPU / 驱动 / CUDA 版本，并为 `UV_TORCH_BACKEND` 推荐一个 uv PyTorch 后端标签（`cpu | cu118 | cu121 | cu124 | cu126 | cu128`）。由 `setup-env` skill 使用；详见 [setup-env.md](setup-env.md)。

### 用法

```bash
bash scripts/detect_hardware.sh             # 人类可读报告
bash scripts/detect_hardware.sh --backend   # 仅打印后端标签
UV_TORCH_BACKEND="$(bash scripts/detect_hardware.sh --backend)" uv sync --python 3.12
```

### 输出

```
gpu=NVIDIA GeForce RTX 4090
driver=550.54.15
cuda=12.4
backend=cu124
```

### 备注

- 无 GPU / `PATH` 中无 `nvidia-smi` → 报告 `backend=cpu`。
- 将驱动支持的最高 CUDA 版本映射到不超过该版本的最高可用 wheel 后端（CUDA 在同一主版本内向后兼容）。
- 只读：不安装任何东西，仅报告与推荐。
