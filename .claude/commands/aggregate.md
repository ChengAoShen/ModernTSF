Aggregate experiment results and optionally plot a bubble chart for the ModernTSF project.

Ask the user:
1. Which dataset to aggregate (e.g. `ETTh1`, `electricity`)
2. Any filters to apply (optional — e.g. `pred_len=96`, `model~Linear`, `mse<=0.5`)
3. Whether to also generate a bubble chart

## Step 1 — Aggregate results

```bash
uv run python tool/aggregate_results.py \
  --dataset <dataset> \
  --output work_dirs/<dataset>/results_all.csv
```

With filters:
```bash
uv run python tool/aggregate_results.py \
  --dataset <dataset> \
  --filter "pred_len=96,model~Linear" \
  --output work_dirs/<dataset>/results_filtered.csv
```

Default output fields: `model, seq_len, pred_len, mse, mae` (perf) + `latency_avg_ms, throughput_samples_sec, total_params, peak_vram_mb` (profile).

## Step 2 — Plot bubble chart (if requested)

```bash
uv run python tool/plot_bubble.py \
  --csv work_dirs/<dataset>/results_all.csv \
  --x latency_avg_ms \
  --y mse \
  --size total_params \
  --size-scale log \
  --x-scale log
```

Common axis choices: `mse`, `mae`, `latency_avg_ms`, `total_params`, `peak_vram_mb`. Output is saved to `work_dirs/plots/bubble_<csv>.svg` by default.

## Notes

- `aggregate_results.py` searches `work_dirs/<dataset>/*/performance.csv` and `profile.csv`.
- Profile data is only available if `evaluation.enable_profile = true` was set in the run config.
- Filter operators: `=`, `!=`, `<`, `>`, `<=`, `>=`, `~` (substring match).
