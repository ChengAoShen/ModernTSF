# 汇总结果

将数据集目录下的 `performance.csv` 与 `profile.csv` 合并为一个 CSV。使用 `run_id` 对齐，performance 为主。

## 用法

```bash
uv run python tool/aggregate_results.py --dataset periodic
```

## 示例

```bash
uv run python tool/aggregate_results.py --dataset periodic --filter "pred_len=96,mse<=0.1,model~Linear"
```

```bash
uv run python tool/aggregate_results.py --dataset periodic --perf-fields "model,seq_len,pred_len,mse,mae" --prof-fields "latency_avg_ms,peak_vram_mb" --output work_dirs/periodic/pred_96.csv
```

## 参数

- `--dataset`：数据集名称（在 `work_dirs` 下）必填。
- `--work-dir`：工作目录根路径（默认 `./work_dirs`）。
- `--output`：输出 CSV 路径（默认 `work_dirs/<dataset>/results_all.csv`）。
- `--filter`：逗号分隔的 AND 过滤条件，支持 `=`, `!=`, `<`, `>`, `<=`, `>=`, `~`（子串匹配）。
- `--perf-fields`：保留 `performance.csv` 的字段。
- `--prof-fields`：保留 `profile.csv` 的字段。
- `--collapse`：TFB 公平性模式。将行折叠为每个 `(model, pred_len)` 一行，用 `--aggregate` 在多个 seed 间聚合指标列，而非输出原始的逐次运行行。默认关闭（保留原始透传）。
- `--aggregate {mean,median,max}`：`--collapse` 在 `(model, pred_len)` 单元内跨 seed/运行聚合指标所用的方式（默认 `mean`）。未设置 `--collapse` 时忽略。
- `--null-threshold`：配合 `--collapse`，排除在超过该比例的 `(pred_len)` 单元上为 NaN/缺失的模型。默认未设置（不排除）；被剔除的模型会记入日志，不会被静默截断。典型值：`0.3`。
- `--metric-cols`：公平性策略聚合并检查空值的指标列（逗号分隔，默认 `mse,mae`）。仅在 `--collapse` 时生效。

## 默认字段

- `--perf-fields`：`model,seq_len,pred_len,mse,mae`
- `--prof-fields`：`latency_avg_ms,throughput_samples_sec,total_params,peak_vram_mb`

## 备注

- 会搜索 `work_dirs/<dataset>/*/performance.csv` 与 `profile.csv`。
- 若其中一个文件缺失，只汇总存在的那一类。
- 指定字段如果不存在会被忽略并提示 warning。
