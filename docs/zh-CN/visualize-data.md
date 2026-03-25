# 数据集可视化

ModernTSF 内置数据可视化脚本：

`tool/visual_data.py`

## 基本用法

```bash
uv run python tool/visual_data.py --config configs/datasets/etth1.toml --split train --num-samples 3 --save work_dirs/plots/etth1.png
```

## 常用参数

- `--config`：TOML 配置路径，可使用完整 run config 或仅含 `[dataset]` 的配置。
- `--split`：`train` / `val` / `test`。
- `--num-samples`：当未指定 `--index` 时绘制的样本数量。
- `--index`：指定绘制的样本索引。
- `--channels`：通道索引列表（逗号分隔）或 `all`。
- `--save`：输出图片路径，默认 `work_dirs/plots/<dataset>_<split>.png`。
- `--show`：弹出窗口显示图像。
- `--seed`：采样随机种子。

## 单通道示例

```bash
uv run python tool/visual_data.py --config configs/datasets/etth1.toml --split train --channels 0 --save work_dirs/plots/etth1_ch0.png
```

## 数据集单独配置

如果配置文件只有 `[dataset]`，脚本会使用 `configs/base.toml` 中的默认 `task` 参数。需要自定义时，可在同一配置中添加 `[task]`。
