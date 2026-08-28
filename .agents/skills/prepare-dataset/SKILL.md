---
name: prepare-dataset
description: Convert or normalize existing time-series files into ModernTSF-ready splits. Use for windowing CSV data, producing NPZ splits, converting traffic bundles, or downloading GIFT-Eval data; not for registering a new loader.
---

# Prepare a dataset

Inspect source layout and destination before writing. For CSV windowing:

```bash
uv run tsf dataset prepare --input-csv <file.csv> --output-dir <dir> \
  --seq-len 96 --label-len 48 --pred-len 96
```

Use `--input-dir` for pre-split CSVs, `uv run tsf dataset convert-traffic --help` for graph data, or `uv run tsf dataset gift-download --help` for GIFT-Eval.

Verify all splits, shapes, requested windows, train-only scaling, and preservation of source files. Stop before replacing an existing output directory without authorization.
