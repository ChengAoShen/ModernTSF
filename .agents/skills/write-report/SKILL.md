---
name: write-report
description: Generate and review a shareable Markdown report from completed ModernTSF results. Use for a benchmark summary, leaderboard narrative, or report artifact; not for raw aggregation alone.
---

# Write a report

```bash
uv run tsf result report --dataset <name> --top 10
```

Use `--pred-len`, `--out`, or `--no-plot` as needed. Read the generated report and verify rankings, metric direction, missing values, counts, and plot references against aggregated data. Deliver the artifact path and scope. If evidence is incomplete, return to experiment or analysis rather than inventing conclusions.
