# 检查配置

预览运行 TOML 的 sweep 展开结果，无需执行任何训练。输出展开后的总运行数，以及数据集、模型、预测步长、随机种子及每个 sweep 参数的唯一值。

## 用法

```bash
uv run python tool/inspect_config.py --config configs/runs/multi_sweep.toml
```

## 输出示例

```
Total runs: 32
Datasets: ETTh1, ETTm1
Models: DLinear, Linear
Pred lens: 96, 192, 336, 720
Seeds: 0, 1
Sweep values:
  extend.datasets: etth1, ettm1
  extend.models: DLinear, Linear
  experiment.random_seed: 0, 1
  task.pred_len: 96, 192, 336, 720
```

`Total runs` 为所有 sweep 轴的乘积（此处为 2 数据集 × 2 模型 × 4 pred\_len × 2 种子 = 32）。`Sweep values` 块列出每个被 sweep 的键及其对应的不同取值；若配置中无 `[sweep]` 节，该块会被省略。

## 参数

- `--config`：运行 TOML 文件路径（必填）。

## 备注

- 会完整展开配置（包括 `extends` 链、`[sweep]` 和 `[sweep.extend]`），但不加载数据也不构建模型。
- 适合在长时间运行前检查 sweep 设置是否正确。
