#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU_IDS="0"

run_config() {
  local config="$1"
  echo "Running: ${config}"
  (cd "${ROOT_DIR}" && CUDA_VISIBLE_DEVICES="${GPU_IDS}" uv run modern-tsf --config "${config}")
}

configs=("configs/runs/sweep_data.toml")

for config in "${configs[@]}"; do
  run_config "${config}"
done
