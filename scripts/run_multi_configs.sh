#!/usr/bin/env bash
set -euo pipefail

# Run one or more ModernTSF experiment configs sequentially.
#
# Usage:
#   bash scripts/run_multi_configs.sh [config ...]
#
# Args:
#   config ...   One or more TOML config paths (relative to repo root).
#                Defaults to configs/runs/run_single_data.toml when none given.
#
# Env:
#   GPU_IDS      GPU id(s) passed via CUDA_VISIBLE_DEVICES (default: 0).
#
# Examples:
#   bash scripts/run_multi_configs.sh
#   GPU_IDS=1 bash scripts/run_multi_configs.sh configs/runs/sweep_data.toml configs/runs/sweep_model.toml

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  sed -n '3,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
  exit 0
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_IDS="${GPU_IDS:-0}"

configs=("$@")
if [[ ${#configs[@]} -eq 0 ]]; then
  configs=("configs/runs/run_single_data.toml")
fi

for config in "${configs[@]}"; do
  echo "Running: ${config} (GPU_IDS=${GPU_IDS})"
  (cd "${ROOT_DIR}" && CUDA_VISIBLE_DEVICES="${GPU_IDS}" uv run modern-tsf --config "${config}")
done
