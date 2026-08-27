---
name: evaluate-gift
description: Prepare and run the repository's GIFT-Eval benchmark workflow. Use when evaluating models on GIFT-Eval datasets or constructing its sweep; not for ordinary local datasets.
---

# Evaluate GIFT-Eval

Inspect `uv run tsf dataset gift-download --help` and obtain only requested data. Then:

```bash
uv run tsf inspect --config configs/runs/gift_eval_sweep.toml
uv run tsf run configs/runs/gift_eval_sweep.toml
```

Confirm dataset versions, horizons, model compatibility, compute budget, and missing-series policy. Aggregate via the result commands and report incomplete cells instead of dropping them silently.
