---
name: report
description: Generate a shareable Markdown benchmark report for a dataset (leaderboard + bubble chart + results table) via `tool/tsf.py report`. Use when the user wants a written report, summary, or scoreboard of experiment results for a dataset after running experiments.
---

## When to use

After running experiments (results land in `work_dirs/<dataset>/<model>/performance.csv`),
use this to produce one shareable Markdown file combining a Top-N leaderboard, a
performance bubble chart, and a results table.

## Command

```bash
# Report for a dataset (default output: work_dirs/<dataset>/report.md)
uv run python tool/tsf.py report --dataset ETTh1

# Filter to one prediction length, show a Top-15 leaderboard, custom output path
uv run python tool/tsf.py report --dataset ETTh1 --pred-len 96 --top 15 --out reports/etth1.md

# Skip the bubble chart
uv run python tool/tsf.py report --dataset ETTh1 --no-plot
```

`--dataset` is the `work_dirs/<dataset>/` subfolder name (the dataset's registry
name key, e.g. `ETTh1`, `cauair_st`).

## What it produces

A Markdown file with:
1. **Leaderboard** — models ranked by mean MSE (lower is better), with MAE; NaN/failed
   runs sink to the bottom.
2. **Bubble chart** — x = MSE, y = MAE, size = `total_params` (embedded image).
   Skipped with a note if there is no `total_params` column (no profiling data).
3. **Results table** — key per-run columns (model, seq_len, pred_len, mse, mae, …).

## Notes

- Reuses `tool/aggregate_results.py` and `tool/plot_bubble.py` under the hood; pure
  standard library otherwise.
- If no results exist for the dataset yet, it writes a stub report telling the user
  to run experiments first (`uv run python tool/tsf.py run <config>`).
- Output lives under `work_dirs/` (gitignored) unless `--out` points elsewhere.

## Reference

See `docs/en/scripts.md` for the full `tsf` command reference, and the `aggregate`
/ `rank` / `plot` skills for the individual building blocks.
