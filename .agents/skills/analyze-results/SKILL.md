---
name: analyze-results
description: Aggregate, filter, rank, compare, plot, and report completed ModernTSF experiment results. Use for exploratory analysis, leaderboards, prediction plots, or a verified shareable report.
---

# Analyze results

Read artifacts, investigate anomalies, choose meaningful comparisons, and write
conclusions with native Agent tools. Use library computations for reproducible
aggregation and protocol checks. The following CLI helpers are optional; do not
require a command just to reason, create a plot, or compose a report. Preserve
source artifacts and validate any computed claims.

```bash
uv run tsf result aggregate --dataset <name> --collapse \
  --aggregate mean --null-threshold 0.3
uv run tsf result rank --help
uv run tsf result plot --help
uv run tsf result predictions --help
uv run tsf result report --help
```

Keep raw and collapsed data distinct. State metric direction, horizon filters, seed aggregation, missing-cell policy, and profile availability. Never compare across incompatible datasets or evaluation protocols.

For a formal report, generate it only after verifying the comparison set and
aggregation policy. Read the artifact back and check rankings, metric direction,
missing values, counts, uncertainty, and plot references against aggregated data.
Deliver the artifact path and scope; do not turn incomplete evidence into a claim.
When a research round exists, append the evidence-backed conclusion and next
decision, then mark it completed, blocked, or stopped as appropriate. Metrics
remain in result artifacts rather than being duplicated as narrative memory.
For published-number replication, return aligned aggregates to
`reproduce-paper-results` so protocol deviations stay attached to the comparison.
