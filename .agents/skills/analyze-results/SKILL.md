---
name: analyze-results
description: Aggregate, filter, rank, compare, or plot completed ModernTSF experiment results. Use for leaderboards, seed aggregation, fairness filtering, performance profiles, and prediction plots; not for writing the final narrative report.
---

# Analyze results

```bash
uv run tsf result aggregate --dataset <name> --collapse \
  --aggregate mean --null-threshold 0.3
uv run tsf result rank --help
uv run tsf result plot --help
uv run tsf result predictions --help
```

Keep raw and collapsed data distinct. State metric direction, horizon filters, seed aggregation, missing-cell policy, and profile availability. Never compare across incompatible datasets or evaluation protocols.

Use `write-report` only after the comparison set and aggregation policy are
verified; keep exploratory plots and intermediate rankings in this skill.
For published-number replication, return aligned aggregates to
`reproduce-paper-results` so protocol deviations stay attached to the comparison.
