#!/usr/bin/env bash
set -euo pipefail

# Aggregate results for a dataset/pred_len then plot a bubble chart.
#
# Usage:
#   bash scripts/aggregate_and_plot.sh [DATASET] [PRED_LEN]
#
# Args:
#   DATASET      Dataset name key (positional or $DATASET env, default: ETTh1).
#   PRED_LEN     Prediction length (positional or $PRED_LEN env, default: 96).
#
# Env:
#   X            Bubble x-axis field (default: latency_avg_ms).
#   Y            Bubble y-axis field (default: mse).
#   SIZE         Bubble size field (default: total_params).
#   OUT_CSV      Aggregated CSV output path
#                (default: work_dirs/${DATASET}/results_all.csv).
#   OUT_SVG      Bubble chart output path
#                (default: work_dirs/plots/bubble_${DATASET}_pl${PRED_LEN}.svg).
#
# Examples:
#   bash scripts/aggregate_and_plot.sh ETTh1 96
#   DATASET=weather PRED_LEN=720 Y=mae bash scripts/aggregate_and_plot.sh

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '3,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

DATASET="${1:-${DATASET:-ETTh1}}"
PRED_LEN="${2:-${PRED_LEN:-96}}"

X="${X:-latency_avg_ms}"
Y="${Y:-mse}"
SIZE="${SIZE:-total_params}"

OUT_CSV="${OUT_CSV:-work_dirs/${DATASET}/results_all.csv}"
OUT_SVG="${OUT_SVG:-work_dirs/plots/bubble_${DATASET}_pl${PRED_LEN}.svg}"

uv run python tool/aggregate_results.py \
  --dataset "${DATASET}" \
  --filter "pred_len=${PRED_LEN}" \
  --perf-fields "model,seq_len,pred_len,mse,mae" \
  --prof-fields "latency_avg_ms,throughput_samples_sec,total_params,peak_vram_mb" \
  --output "${OUT_CSV}"

uv run python tool/plot_bubble.py \
  --csv "${OUT_CSV}" \
  --x "${X}" \
  --y "${Y}" \
  --size "${SIZE}" \
  --size-scale log \
  --x-scale log \
  --y-scale log \
  --color-by model \
  --label-by model \
  --output "${OUT_SVG}"
