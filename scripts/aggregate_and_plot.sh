#!/usr/bin/env bash
set -euo pipefail

DATASET="trend"
PRED_LEN="720"

OUT_CSV="work_dirs/${DATASET}/results_all.csv"
OUT_SVG="work_dirs/plots/bubble_${DATASET}_pl${PRED_LEN}.svg"

uv run python tool/aggregate_results.py \
  --dataset "${DATASET}" \
  --filter "pred_len=${PRED_LEN}" \
  --perf-fields "model,seq_len,pred_len,mse,mae" \
  --prof-fields "latency_avg_ms,throughput_samples_sec,total_params,peak_vram_mb" \
  --output "${OUT_CSV}"

uv run python tool/plot_bubble.py \
  --csv "${OUT_CSV}" \
  --x latency_avg_ms \
  --y mse \
  --size total_params \
  --size-scale log \
  --x-scale log \
  --y-scale log \
  --color-by model \
  --label-by model \
  --output "${OUT_SVG}"
