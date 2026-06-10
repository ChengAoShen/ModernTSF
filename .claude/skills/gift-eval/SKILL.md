---
name: gift-eval
description: Download GIFT-EVAL datasets from HuggingFace and run the full 53-dataset GIFT-EVAL benchmark sweep. Use when the user wants to run, reproduce, or evaluate against the GIFT-EVAL benchmark.
---

## When to use / what to ask

Ask the user:

1. **Where to store the data?** Default `~/.cache/gift_eval`.
2. **All 53 datasets or a subset?** Omit `--datasets` for all; `--list` shows the `base/freq` names.
3. **Already downloaded?** Use `--link-only` to skip the download and just create the symlink.

## Commands

```bash
# Step 1 — download + symlink (dataset/gift_eval -> <output-dir>)
uv run python tool/gift_eval_download.py [--output-dir DIR] [--datasets electricity/15T m4_monthly ...] [--link-only] [--list]

# Step 2 — run the benchmark sweep (preview first with the `inspect` skill)
uv run modern-tsf --config configs/runs/gift_eval_sweep.toml

# Step 3 — aggregate + plot per dataset
uv run python tool/tsf.py aggregate-plot --dataset <name> --pred-len <len>
```

## Notes

- The `dataset/gift_eval` symlink is what TOML configs reference (`root_path = "./dataset/gift_eval"`); re-running with a different `--output-dir` updates it.
- Unknown names passed to `--datasets` warn but do not abort.
- Results land in `work_dirs/<dataset>/<model>/performance.csv`.

## Reference

Dataset list, config layout, and troubleshooting: `docs/en/gift-eval.md`.
