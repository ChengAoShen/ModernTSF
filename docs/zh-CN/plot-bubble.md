# 气泡图

从 CSV 文件读取数据，指定 x、y 和大小字段绘制气泡图，默认按模型着色。典型输入为 `aggregate_results.py` 生成的 `results_all.csv`。

## 用法

```bash
uv run python tool/plot_bubble.py \
    --csv work_dirs/<dataset>/results_all.csv \
    --x <x字段> \
    --y <y字段> \
    --size <大小字段>
```

## 参数

| 标志 | 必填 | 默认值 | 说明 |
|---|---|---|---|
| `--csv` | 是 | — | 输入 CSV 路径 |
| `--x` | 是 | — | x 轴字段 |
| `--y` | 是 | — | y 轴字段 |
| `--size` | 是 | — | 气泡大小字段 |
| `--size-scale` | 否 | `linear` | 大小值归一化前的变换方式：`linear`、`sqrt` 或 `log` |
| `--x-scale` | 否 | `linear` | x 轴坐标系：`linear` 或 `log` |
| `--y-scale` | 否 | `linear` | y 轴坐标系：`linear` 或 `log` |
| `--color-by` | 否 | `model` | 用于分配气泡颜色的字段 |
| `--label-by` | 否 | `model` | 用于标注每个气泡的字段 |
| `--no-labels` | 否 | 关 | 禁用气泡文字标注 |
| `--legend` | 否 | 关 | 显示图例 |
| `--output` | 否 | `work_dirs/plots/bubble_<csv>.png` | 输出图片路径 |
| `--show` | 否 | 关 | 打开交互式绘图窗口 |
| `--title` | 否 | 自动 | 图标题；默认为 `<x> vs <y> \| size: <size> (<size-scale>)` |

## 缩放选项

### `--size-scale`

控制原始大小值在线性映射到点面积区间 `[30, 300]` 之前的变换方式。

| 取值 | 效果 | 适用场景 |
|---|---|---|
| `linear` | 不做变换，直接使用原始值 | 大小字段范围小且分布均匀 |
| `sqrt` | `value ^ 0.5`，压缩大值、展开小值 | 参数量等跨度约一个数量级的字段 |
| `log` | `log10(value)`，对大值进行强力压缩 | 跨越多个数量级的字段（如 `total_params` 从 10 K 到 100 M） |

选择 `log` 时，非正数值的行会被丢弃；选择 `sqrt` 时，负数值的行会被丢弃。

### `--x-scale` / `--y-scale`

设置 matplotlib 坐标轴的缩放方式。`log` 使用以 10 为底的对数轴。在 `log` 轴上，非正数值的行会在绘图前被移除。

## 常用字段参考

以下字段来自 `aggregate_results.py` 生成的 CSV（性能数据与性能分析数据合并后）。

| 字段 | 来源 | 典型用途 |
|---|---|---|
| `mse` | `performance.csv` | 精度 — x 或 y 轴 |
| `mae` | `performance.csv` | 精度 — x 或 y 轴 |
| `pred_len` | `performance.csv` | 分组轴 |
| `latency_avg_ms` | `profile.csv` | 推理耗时 — x 轴 |
| `throughput_samples_sec` | `profile.csv` | 吞吐量 — x 轴 |
| `total_params` | `profile.csv` | 模型参数量 — 大小或轴 |
| `peak_vram_mb` | `profile.csv` | 显存占用 — 大小或轴 |

## 示例

### 精度 vs 推理耗时，气泡大小 = 参数量

```bash
uv run python tool/plot_bubble.py \
    --csv work_dirs/ETTh1/results_all.csv \
    --x latency_avg_ms \
    --y mse \
    --size total_params \
    --size-scale log \
    --x-scale log \
    --title "ETTh1: accuracy vs latency"
```

两轴均使用 `log` 缩放，因为延迟和 MSE 可能相差一个数量级。`--size-scale log` 可避免单个大模型在视觉上过于突出。

### MSE vs MAE，气泡大小 = 显存，保存到自定义路径

```bash
uv run python tool/plot_bubble.py \
    --csv work_dirs/weather/results_all.csv \
    --x mse \
    --y mae \
    --size peak_vram_mb \
    --size-scale sqrt \
    --legend \
    --output work_dirs/weather/mse_mae_vram.svg
```

`--size-scale sqrt` 是折中方案：既能降低高显存模型的视觉主导性，又不像 `log` 那样激进。

### 按预测步长分组而非按模型分组

```bash
uv run python tool/plot_bubble.py \
    --csv work_dirs/ETTh1/results_all.csv \
    --x total_params \
    --y mse \
    --size latency_avg_ms \
    --color-by pred_len \
    --label-by model \
    --legend
```

## 备注

- 会从字符串中自动提取数值（例如 `"2.5 ms"` → `2.5`）。
- 默认在每个气泡附近显示标签并隐藏图例。
- 不超过 20 个类别使用 `tab20` 调色板；21–60 个使用 `turbo`；超过 60 个使用 `hsv`。
- 如果输出目录不存在，会自动创建。
- 对于 `log` 缩放（轴或大小），非正数值的行会被静默移除。
