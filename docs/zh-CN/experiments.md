# 一键实验

ModernTSF 把消融、超参数搜索、case 可视化都做成了单条命令。前两者由运行配置里的 **sweep** 机制驱动（在加载时展开为笛卡尔积——见 [configs.md](configs.md)）；第三个是一个小绘图工具。

启动前可先预览 sweep 规模（运行数、数据集、模型、预测长度）：

```bash
uv run python tool/inspect_config.py --config configs/runs/<your_sweep>.toml
```

运行 sweep（单进程）或在指定 GPU 上批量跑多个配置：

```bash
uv run modern-tsf --config configs/runs/<your_sweep>.toml
# 或在选定 GPU 上顺序跑：
uv run python tool/tsf.py run configs/runs/<your_sweep>.toml --gpus 0
```

每次运行都会向 `work_dirs/<dataset>/<model>/performance.csv` 写一行；用 `tool/aggregate_results.py` / `tool/rank_models.py` 汇总与排名。

---

## 1. 消融实验

消融即开关模型的**组件**并比较效果。由于每个组件都是 `[model.params]` 里的一个字段，用 `[sweep]` 扫这些字段即可：

```toml
# configs/runs/ablation_dlinear.toml
extends = ["../base.toml", "../datasets/etth1.toml", "../models/DLinear.toml"]

[sweep]
# 每个列出的值都展开成一次独立运行（笛卡尔积）。
model.params.individual = [true, false]      # 逐通道预测头 开/关
model.params.kernel_size = [13, 25, 49]      # 分解窗口

[sweep.task]
pred_len = [96, 336]
```

`inspect_config.py` 会报告为 `2 × 3 × 2 = 12` 次运行。也可以用 `[sweep.extend]` **整体替换模型变体**（如 DLinear vs NLinear vs RLinear）做消融——见 `configs/runs/sweep_model.toml`。

## 2. 超参数实验

机制完全相同——把开关换成数值/结构超参即可：

```toml
# configs/runs/hparam_patchtst.toml
extends = ["../base.toml", "../datasets/etth1.toml", "../models/PatchTST.toml"]

[sweep]
model.params.d_model = [128, 256, 512]
model.params.n_heads = [4, 8]
training.optimizer.params.lr = [0.0001, 0.0005]

[sweep.task]
pred_len = [96, 192, 336, 720]
```

多数据集 / 多模型 / 多种子网格见 `configs/runs/multi_sweep.toml`（`数据集 × 模型 × 种子 × 预测长度`）和 `configs/runs/sweep_data.toml`（一个模型跑全部数据集）。设 `[evaluation] enable_profile = true` 可同时记录每次运行的参数量/MACs。

## 3. Case 可视化

把模型在若干测试窗口上的预测与真值画在一起。先训练模型（产生 checkpoint），然后：

```bash
# 训练（写出 work_dirs/<dataset>/<model>/checkpoints/<run_id>/best_checkpoint.pth）
uv run modern-tsf --config configs/runs/run_single_data.toml

# 可视化——自动找该 (数据集, 模型) 最新的 checkpoint
uv run python tool/visualize_predictions.py \
    --config configs/runs/run_single_data.toml --num-samples 4 --channel -1
```

输出 `work_dirs/<dataset>/<model>/cases.png`，每个样本画出：输入历史、真实未来、预测未来（单通道/节点）。参数：`--num-samples N`、`--channel I`（`-1` = 最后一个通道）、`--checkpoint PATH`（指定 checkpoint）、`--out PATH`。所有模型通用，包括时空/图模型与协变量模式（用 `--channel` 选节点）。

---

另见：[configs.md](configs.md)（sweep 语义）、[aggregate-results.md](aggregate-results.md)、[rank-models.md](rank-models.md)、[plot-bubble.md](plot-bubble.md)。
