# 统一工具入口（`tsf`）与脚本

ModernTSF 提供单一统一入口 —— `tool/tsf.py` —— Agent（或你）可用它驱动所有常见操作。
它是纯标准库（`argparse` + `concurrent.futures` + `subprocess`），通过 `uv` 运行，
需要时并发。旧的 `run_multi_configs.sh` 与 `aggregate_and_plot.sh` 胶水脚本已退役，
由 `tsf run` 与 `tsf aggregate-plot` 取代。

```bash
uv run python tool/tsf.py <command> [args...]
uv run python tool/tsf.py --help                 # 列出所有命令
uv run python tool/tsf.py <command> --help       # 某命令自身的参数
```

---

## 脚手架

| 命令 | 作用 |
|---|---|
| `new-model` | 生成模型包 + `schema.py` + `registry.py` + config + smoke 运行配置，并插入 `MODEL_NAME_MAP` 条目。 |
| `new-dataset` | 生成数据集（`--pattern custom` / `presplit` = 仅 config；`single` = 完整代码 + 接线）。 |

```bash
# 普通 (B, T, C) 预测器，带两个超参数
uv run python tool/tsf.py new-model --name MyModel --params "enc_in:int,hidden:int=128"

# 节点结构的图 / 时空模型（读取 params["adj_mx"]）
uv run python tool/tsf.py new-model --name MyGraphNet --graph --params "enc_in:int,hidden:int=64"

# 仅 config 的自定义 CSV 数据集
uv run python tool/tsf.py new-dataset --name my_csv --pattern custom \
    --root-path ./dataset/my_csv --data-path my_csv.csv --target OT
```

`new-model` 之后，把架构实现填进生成的 `model.py` 的 `forward`，然后验证（见下）。

---

## 验证与运行（并发）

| 命令 | 作用 |
|---|---|
| `smoke` | 端到端运行 smoke 配置并报告 PASS/FAIL。`--all`、`--model <Name>` 或 `--config <paths>`；`--jobs N`（默认 `min(8, cpu)`）。 |
| `run` | 运行实验配置。`--jobs N`（默认 1）并行运行；`--gpus 0,1` 在各 job 间轮询 `CUDA_VISIBLE_DEVICES`。 |

```bash
# 验证单个新模型
uv run python tool/tsf.py smoke --model MyModel

# 8 路并发验证仓库内全部模型
uv run python tool/tsf.py smoke --all --jobs 8

# 两个 sweep 配置在两块 GPU 上并行
uv run python tool/tsf.py run configs/runs/sweep_data.toml configs/runs/sweep_model.toml --jobs 2 --gpus 0,1
```

任一配置失败时 `smoke` 以非零退出，因此它也可直接当作 CI 关卡。

---

## 结果与绘图

| 命令 | 作用 |
|---|---|
| `report` | 生成可分享的 Markdown 报告（排行榜 + 气泡图 + 结果表）。`--dataset`、`--pred-len`、`--top`、`--out`、`--no-plot`。 |
| `aggregate-plot` | 一步完成：聚合某数据集结果 + 绘制气泡图（取代 `aggregate_and_plot.sh`）。`--dataset`、`--pred-len`、`--x/--y/--size`、`--out-csv/--out-svg`。 |
| `aggregate` | → `tool/aggregate_results.py` |
| `rank` | → `tool/rank_models.py` |
| `plot` | → `tool/plot_bubble.py` |
| `characteristics` | → `tool/dataset_characteristics.py` |
| `visualize` | → `tool/visual_data.py` |
| `predictions` | → `tool/visualize_predictions.py` |
| `inspect` | → `tool/inspect_config.py` |

```bash
uv run python tool/tsf.py aggregate-plot --dataset ETTh1 --pred-len 96
uv run python tool/tsf.py rank --dataset ETTh1
uv run python tool/tsf.py inspect --config configs/runs/multi_sweep.toml
```

---

## 数据准备

| 命令 | 作用 |
|---|---|
| `pre-process` | → `tool/pre_process.py`（CSV → 预切窗 `.npz`） |
| `convert-traffic` | → `tool/convert_traffic.py`（取值数组 + 邻接 → 节点包） |
| `gift-download` | → `tool/gift_eval_download.py`（下载 GIFT-EVAL 数据集） |

转发类命令会原样接受底层工具的参数，因此
`tsf aggregate --dataset ETTh1` 等价于 `python tool/aggregate_results.py --dataset ETTh1`。

---

## `scripts/detect_hardware.sh`

唯一保留的 shell 脚本：探测 GPU / 驱动 / CUDA 版本，并为 `UV_TORCH_BACKEND` 推荐一个
uv PyTorch 后端标签（`cpu | cu118 | cu121 | cu124 | cu126 | cu128`）。供 `setup-env`
skill 使用；见 [setup-env.md](setup-env.md)。

```bash
bash scripts/detect_hardware.sh             # 人类可读报告
bash scripts/detect_hardware.sh --backend   # 仅打印后端标签
UV_TORCH_BACKEND="$(bash scripts/detect_hardware.sh --backend)" uv sync --python 3.12
```

输出：

```
gpu=NVIDIA GeForce RTX 4090
driver=550.54.15
cuda=12.4
backend=cu124
```

- 无 GPU / `PATH` 上无 `nvidia-smi` → 报告 `backend=cpu`。
- 将驱动支持的最高 CUDA 版本映射到 ≤ 该版本的最高可用 wheel 后端。
- 只读：从不安装任何东西，只做报告与推荐。
