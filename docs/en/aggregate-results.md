# Aggregate results

Combine `performance.csv` and `profile.csv` under a dataset directory into a single CSV. Rows are merged by `run_id`, with performance as the primary source.

## Usage

```bash
uv run python tool/aggregate_results.py --dataset periodic
```

## Examples

```bash
uv run python tool/aggregate_results.py --dataset periodic --filter "pred_len=96,mse<=0.1,model~Linear"
```

```bash
uv run python tool/aggregate_results.py --dataset periodic --perf-fields "model,seq_len,pred_len,mse,mae" --prof-fields "latency_avg_ms,peak_vram_mb" --output work_dirs/periodic/pred_96.csv
```

## Arguments

- `--dataset`: dataset name under `work_dirs` (required).
- `--work-dir`: root work directory (default: `./work_dirs`).
- `--output`: output CSV path (default: `work_dirs/<dataset>/results_all.csv`).
- `--filter`: comma-separated AND filters. Operators: `=`, `!=`, `<`, `>`, `<=`, `>=`, `~` (substring).
- `--perf-fields`: fields to keep from `performance.csv`.
- `--prof-fields`: fields to keep from `profile.csv`.

## Defaults

- `--perf-fields`: `model,seq_len,pred_len,mse,mae`
- `--prof-fields`: `latency_avg_ms,throughput_samples_sec,total_params,peak_vram_mb`

## Notes

- Searches `work_dirs/<dataset>/*/performance.csv` and `profile.csv`.
- If one file is missing, only the available one is aggregated.
- Missing fields are ignored with a warning.
