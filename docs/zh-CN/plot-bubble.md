# 气泡图

从 CSV 读取数据，指定 x、y 和大小字段绘制气泡图，默认按模型着色。

## 用法

```bash
uv run python tool/plot_bubble.py --csv work_dirs/periodic/results_all.csv --x mse --y mae --size total_params
```

## 示例

```bash
uv run python tool/plot_bubble.py --csv work_dirs/periodic/results_all.csv --x latency_avg_ms --y mse --size total_params --size-scale log --x-scale log
```

## 参数

- `--csv`：输入 CSV 路径。
- `--x`：x 轴字段。
- `--y`：y 轴字段。
- `--size`：气泡大小字段。
- `--size-scale`：`linear` / `sqrt` / `log`（默认 `linear`）。
- `--x-scale`：`linear` / `log`（默认 `linear`）。
- `--y-scale`：`linear` / `log`（默认 `linear`）。
- `--color-by`：颜色分组字段（默认 `model`）。
- `--label-by`：标注字段（默认 `model`）。
- `--no-labels`：关闭点标注。
- `--legend`：显示图例。
- `--output`：输出图片路径（默认 `work_dirs/plots/bubble_<csv>.svg`）。
- `--show`：显示窗口。
- `--title`：图标题（可选）。

## 备注

- 会从字符串中自动提取数值（例如 `"2.5 ms"`）。
- 默认在每个气泡附近显示标签并隐藏图例。
- `log` 缩放会剔除非正数值的行。
