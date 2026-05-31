# 参数说明

本文件说明各 TOML 配置段与字段含义。默认值来源于 `configs/base.toml`，字段定义来源于 `src/benchmark/config/schema/`。

## [experiment]

- `description`（str）：实验描述文本。
- `random_seed`（int）：随机种子，用于复现。
- `work_dir`（str）：输出目录根路径。

### [experiment.runtime]

- `device`（str）：运行设备，常用 `"cuda"` 或 `"cpu"`。
- `use_multi_gpu`（bool）：是否启用多 GPU。
- `device_ids` / `gpus`（list[int]）：GPU id 列表，支持 `gpus` 别名。
- `amp`（bool）：是否启用自动混合精度。
- `num_workers`（int）：DataLoader 线程数。

## [task]

- `seq_len`（int）：输入序列长度。
- `label_len`（int）：解码器 warm-up 长度（部分模型使用）。
- `pred_len`（int）：预测长度。
- `features`（str）：`"M"`、`"S"` 或 `"MS"`。
  - `M`：多变量输入与输出。
  - `S`：单变量目标。
  - `MS`：多变量输入、单变量输出。
- `inverse`（bool）：是否对输出进行反归一化（如果数据集支持）。

## [training]

- `epochs`（int）：训练轮数。
- `batch_size`（int）：批大小。
- `loss`（str）：loss 名称，通过 `LOSS_NAME_MAP` 解析。
- `loss_params`（dict）：传给 loss 构造器的参数，如 `reduction`。
- `patience`（int）：早停耐心轮数。

### [training.optimizer]

- `name`（str）：优化器名称，如 `Adam`。
- `lr`（float）：学习率。
- `weight_decay`（float）：权重衰减。
- `lradj`（str）：学习率调度策略名称（如使用）。
- `params`（dict）：额外优化器参数。

### [training.tricks]（可选——默认全部禁用；不写该段则行为不变）

可插拔训练 callback（`src/benchmark/runner/callbacks.py`）：

- `grad_clip_norm`（float）：优化器步进前裁剪梯度范数。
- `grad_clip_norm_type`（float）：裁剪范数类型（默认 2）。
- `grad_accum_steps`（int）：跨 N 个 micro-batch 累积梯度（更大有效 batch）。
- `[training.tricks.curriculum]`：`enabled`/`warmup_epochs`/`step_size`/`cl_epochs`——逐步增长监督预测步长（BasicTS 方案），上限 `pred_len`。

**辅助损失**：若模型暴露 `self.aux_loss`（或 `last_moe_loss`/`last_aux_loss`）为有限标量张量，trainer 会自动加到训练损失（无则不影响）。适用于 MoE 平衡/KL/正则项（如 Pathformer、TimeFilter）。

性能记录另含 `fit_time` 与 `inference_time`。

### [training.checkpoint]

- `strategy`（str）：保存策略，如 `"best"`。
- `save_k`（int）：保留 checkpoint 数量。

## [dataset]

- `name`（str）：数据集名称，需在 `DATASET_NAME_MAP` 注册。
- `alias`（str | None）：可选的显示别名，用于 CSV 汇总与日志输出（如 `"gift_eval/bizitobs_application"`）。默认为 `null`（使用 `name` 值）。
- `root_path`（str）：数据根目录。默认值：`"./data/"`。
- `data_path`（str）：相对于 `root_path` 的数据文件名。预切分与预处理数据集设为 `""`。
- `params`（dict）：数据集参数，需通过数据集 schema 校验。

### 通用数据集参数

多数单文件数据集支持：

- `target`（str）：目标列名或索引。
- `scale`（bool）：是否应用 `StandardScaler`。默认值：`true`。
- `split_ratio`（list[float]）：训练/验证/测试比例（比例或绝对值），默认值因数据集而异。
- `norm_each_channel`（bool，默认 `false`）：在训练切分上按通道计算 mean/std，而非共享 scaler（opt-in，默认关=行为不变）。
- `target_channel`（int|null，默认 `null`）：将归一化/逆变换锚定到该通道，使目标与协变量通道使用独立统计（用于协变量任务模式）。

### 数据集特有参数

`ETT`（`configs/datasets/etth1.toml` 等）

- 只需通用参数，数据从 CSV 读取。
- 默认 `split_ratio`：`[12.0, 4.0, 4.0]`（月份，与原论文切分一致）。

`traffic` / `weather` / `electricity`

- 只需通用参数，CSV 必须包含 `date` 列用于时间特征。
- 默认 `split_ratio`：`[0.7, 0.1, 0.2]`。

`solar`

- 只需通用参数，数据来自文本文件，非 CSV。
- 默认 `split_ratio`：`[0.7, 0.1, 0.2]`。

`periodic`（合成数据集 — 按下列参数创建 `configs/datasets/periodic.toml`）

- `target`（str）：生成时不使用，但 schema 要求填写。默认值：`"OT"`。
- `scale`（bool）：是否缩放。默认值：`true`。
- `split_ratio`（list[float]）：切分比例。默认值：`[0.7, 0.1, 0.2]`。
- `channel_number`（int）：通道数。默认值：`1`。
- `num_samples`（int）：独立样本数。默认值：`1024`。
- `period`（int）：周期长度（时间步）。默认值：`24`。
- `noise_std`（float）：高斯噪声标准差。默认值：`0.1`。
- `amplitude_range`（list[float]）：幅度范围。默认值：`[0.5, 1.5]`。
- `phase_range`（list[float]）：相位范围（弧度）。默认值：`[0.0, 6.283...]`（0 至 2π）。
- `cycle_start_mode`（str）：起始周期模式（如 `"random"`）。默认值：`"random"`。
- `random_phase`（bool）：是否随机相位。默认值：`true`。

`trend`（合成数据集 — 按下列参数创建 `configs/datasets/trend.toml`）

- `target`（str）：生成时不使用。默认值：`"OT"`。
- `scale`（bool）：默认值：`true`。
- `split_ratio`（list[float]）：默认值：`[0.7, 0.1, 0.2]`。
- `channel_number`（int）：通道数。默认值：`1`。
- `num_samples`（int）：独立样本数。默认值：`1024`。
- `degree_min`（int）：多项式最低次数。默认值：`2`。
- `degree_max`（int）：多项式最高次数。默认值：`6`。
- `coeff_range`（list[float]）：系数采样范围。默认值：`[-0.8, 0.8]`。
- `noise_std`（float）：高斯噪声标准差。默认值：`0.1`。
- `normalize_t`（bool）：是否将时间轴归一化到 `[0, 1]`。默认值：`true`。

`presplit`

- `target`（str）：目标列名。
- `scale`（bool）：是否缩放（scaler 始终在 `train.csv` 上拟合）。
- 无需 `split_ratio`，文件夹须包含 `train.csv`、`val.csv`、`test.csv`。
- `root_path` 指向包含三个文件的文件夹，`data_path` 设为空字符串。

示例配置：

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

- 无 `[dataset.params]` 字段——所有窗口化由 `tool/pre_process.py` 处理。
- `root_path` 指向存放 `.npz` 文件的目录。
- 设 `data_path = ""`。

示例配置：

```toml
[dataset]
name = "pre_processed"
root_path = "./dataset/my_dataset_npy"
data_path = ""
```

`gift_eval`

- `scale`（bool）：是否应用 `StandardScaler`（在训练数据上拟合）。默认值：`true`。
- `windows`（int | None）：滚动测试窗口数。`null` 表示按 GIFT-EVAL 协议自动计算。默认值：`null`。
- 在 `[dataset]` 层用 `alias` 设置 CSV 汇总中的可读名称。

示例配置：

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

- `name`（str）：模型名称，需在 `MODEL_NAME_MAP` 注册。
- `params`（dict）：模型参数，需通过模型 schema 校验。

具体参数以 `src/models/<model>/schema.py` 为准。

## [evaluation]

- `metrics`（list[str]）：指标名称，通过 `METRIC_NAME_MAP` 解析。所有指标始终都会计算，此列表仅决定哪些写入 `performance.csv`。Pydantic schema 默认值为 `["mae", "mse", "rmse", "mape", "mspe"]`，但 `configs/base.toml` 将其扩展为 `["mae", "mse", "rmse", "mape", "mspe", "corr", "rse", "wape", "smape"]`（因此所有继承 `base.toml` 的运行都会记录全部九项）。显式加入 `"mase"` 可一并记录。
- `enable_profile`（bool）：是否在评估后运行模型 profiler。默认值：`false`。
- `strategy`（`"fixed"` | `"rolling"`）：评估策略。默认 `"fixed"`——历史的固定窗口评估，对测试 `DataLoader` 遍历一次。`"rolling"` 启用 TFB 风格的滚动预测（见下文）。省略时行为与之前完全一致。

### 滚动预测（`strategy = "rolling"`）

当 `strategy = "rolling"` 时，评估器在测试集上滚动：取 `seq_len` 输入窗口，预测 `pred_len` 步，按 `stride` 前移窗口，重复至多 `num_rollings` 次（或直到测试数据耗尽），再用 `collect_metrics` 计算相同指标。滚动子配置位于 `[evaluation.rolling]`：

- `horizon`（int | null）：每次滚动评分的预测步数。`null`（默认）使用完整 `pred_len`；并被限制到 `pred_len`（模型每次只输出 `pred_len` 步）。
- `stride`（int）：两次滚动之间输入窗口前移的步数。默认 `1`。
- `num_rollings`（int | null）：最大滚动次数。`null`（默认）滚动到测试数据耗尽为止。

滚动运行会在 `performance.csv` 中追加一列 `eval_strategy=rolling`；固定运行保持 CSV 表头不变。

```toml
[evaluation]
strategy = "rolling"

[evaluation.rolling]
horizon = 96
stride = 1
num_rollings = 100
```

### 可用指标

| 名称 | 说明 |
|---|---|
| `mae` | 平均绝对误差 |
| `mse` | 均方误差 |
| `rmse` | 均方根误差 |
| `mape` | 平均绝对百分比误差 |
| `mspe` | 均方百分比误差 |
| `corr` | 逐通道 Pearson 相关系数均值 |
| `rse` | 相对平方误差 |
| `wape` | 加权绝对百分比误差（尺度无关） |
| `smape` | 对称平均绝对百分比误差 |
| `mase` | 平均绝对缩放误差（需手动开启；评估窗口 naive 基线——见注） |

> `mase` 可用但不在默认列表：评估时只有预测窗口可用，naive 基线由测试目标自身构造（非样本内历史），数值与教科书的样本内缩放 MASE 不可比。需要时在 `[evaluation] metrics` 显式加入。

### 可用损失

通过 `loss` 从 `LOSS_NAME_MAP` 解析。标准：`mae`、`mse`（+ `freq_mae`、`freq_weighted_mae`）。**Masked** 变体 `masked_mae`、`masked_mse`、`masked_rmse` 会忽略可选 `targets_mask`（1=有效，0=忽略）标记的位置，并按 mask 均值归一化（BasicTS 约定）使损失不受有效点数量影响；无 mask 时等同普通版本。适用于交通/缺失值预测。

### 邻接归一化（图模型）

节点结构化数据集向模型工厂注入原始 `adj_mx`。在 `[dataset.params]` 设 `adj_norm = "<scheme>"` 可先归一化，`<scheme>` ∈ `sym_norm_lap` | `scaled_laplacian` | `gcn` | `transition` | `reverse_transition`（见 `src/models/_external/adj_norm.py`）。默认（不设）注入原始矩阵。

### 性能分析（`enable_profile = true`）

启用后，profiler 在评估后在测试 `DataLoader` 上运行，每次运行写出两个文件：

- `work_dirs/<dataset>/<model>/profiles/<run_id>.txt` — 可读报告，包含三个部分：架构/参数摘要（通过 `torchinfo`）、FLOPs（通过 `fvcore`，单位为百万 MACs）、CUDA 延迟/吞吐（10 次预热后 50 次前向计时，CPU 上跳过）。
- `work_dirs/<dataset>/<model>/profile.csv` — 每次运行追加一行的结构化 CSV。

`profile.csv` 各列说明：

| 列名 | 说明 |
|---|---|
| `total_params` | 总参数量 |
| `trainable_params` | 可训练参数量 |
| `non_trainable_params` | 不可训练参数量 |
| `total_mult_adds_mb` | 总乘加次数（MB，来自 torchinfo） |
| `total_macs_m` | 总 MACs（百万，来自 fvcore） |
| `dynamic_vram_mb` | 前向传播动态 VRAM 占用（MB） |
| `peak_vram_mb` | 总峰值 VRAM（MB） |
| `reserved_vram_mb` | 总预留 VRAM（MB） |
| `latency_avg_ms` | 平均前向延迟（ms） |
| `throughput_samples_sec` | 吞吐量（样本/秒） |

`torchinfo` 与 `fvcore` 均为可选依赖。未安装时对应字段从报告中省略（不报错）。在 CPU 上运行时，CUDA 延迟列省略。

## [sweep]

用于配置 sweep，支持点号路径或嵌套表：

```toml
[sweep]
experiment.random_seed = [0, 1, 2]

[sweep.task]
pred_len = [96, 192, 336, 720]
```

所有组合将展开为多个独立实验并顺序执行。
