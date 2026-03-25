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

### [training.checkpoint]

- `strategy`（str）：保存策略，如 `"best"`。
- `save_k`（int）：保留 checkpoint 数量。

## [dataset]

- `name`（str）：数据集名称，需在 `DATASET_NAME_MAP` 注册。
- `root_path`（str）：数据根目录。
- `data_path`（str）：数据文件名（预切分数据集设为空字符串）。
- `params`（dict）：数据集参数，需通过数据集 schema 校验。

### 通用数据集参数

多数数据集支持：

- `target`（str）：目标列名或索引。
- `scale`（bool）：是否缩放。
- `split_ratio`（list[float]）：训练/验证/测试比例。

### 数据集特有参数

`periodic`（合成数据集 — 按下列参数创建 `configs/datasets/periodic.toml`）

- `channel_number`（int）：通道数。
- `num_samples`（int）：样本数量（独立序列数）。
- `period`（int）：周期长度（时间步）。
- `noise_std`（float）：高斯噪声标准差。
- `amplitude_range`（list[float]）：幅度范围。
- `phase_range`（list[float]）：相位范围（弧度）。
- `cycle_start_mode`（str）：起始周期模式（如 `"random"`）。
- `random_phase`（bool）：是否随机相位。

`ETT`（`configs/datasets/etth1.toml` 等）

- 只需通用参数，数据从 CSV 读取并按原论文比例切分。

`traffic` / `weather` / `electricity`

- 只需通用参数，CSV 必须包含 `date` 列用于时间特征。

`solar`

- 只需通用参数，数据来自文本文件。

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

## [model]

- `name`（str）：模型名称，需在 `MODEL_NAME_MAP` 注册。
- `params`（dict）：模型参数，需通过模型 schema 校验。

具体参数以 `src/models/<model>/schema.py` 为准。

## [evaluation]

- `metrics`（list[str]）：指标名称，通过 `METRIC_NAME_MAP` 解析。
- `enable_profile`（bool）：是否开启 profile。

## [sweep]

用于配置 sweep，支持点号路径或嵌套表：

```toml
[sweep]
experiment.random_seed = [0, 1, 2]

[sweep.task]
pred_len = [96, 192, 336, 720]
```

所有组合将展开为多个独立实验并顺序执行。
