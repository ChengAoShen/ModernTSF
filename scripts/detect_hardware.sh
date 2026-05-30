#!/usr/bin/env bash
#
# Detect GPU / CUDA on this machine and recommend a uv PyTorch backend tag for
# UV_TORCH_BACKEND (cpu | cu118 | cu121 | cu124 | cu126 | cu128).
#
# Usage:
#   bash scripts/detect_hardware.sh             # human-readable report
#   bash scripts/detect_hardware.sh --backend   # print only the backend tag
#   UV_TORCH_BACKEND="$(bash scripts/detect_hardware.sh --backend)" uv sync
#
# Note: prefer `UV_TORCH_BACKEND=auto uv sync` — uv detects the driver itself.
# This script is for visibility and for the explicit-override path.
set -euo pipefail

MODE="${1:-report}"

# No NVIDIA tooling → CPU build.
if ! command -v nvidia-smi >/dev/null 2>&1; then
  if [ "$MODE" = "--backend" ]; then echo "cpu"; else
    echo "gpu=none"
    echo "driver=none"
    echo "cuda=none"
    echo "backend=cpu"
  fi
  exit 0
fi

gpu="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd';' - || true)"
driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 || true)"
# Max CUDA version the driver supports (top-right of `nvidia-smi`).
cuda="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1 || true)"

# Map driver CUDA → nearest available PyTorch wheel backend (≤ driver CUDA).
# CUDA minor versions are backward compatible within a major, so a newer driver
# can run an older cuXXX wheel.
case "${cuda:-}" in
  13.*)               backend="cu128" ;;
  12.8*|12.9*)        backend="cu128" ;;
  12.6*|12.7*)        backend="cu126" ;;
  12.4*|12.5*)        backend="cu124" ;;
  12.1*|12.2*|12.3*)  backend="cu121" ;;
  11.8*|11.9*|12.0*)  backend="cu118" ;;
  *)                  backend="cpu"   ;;
esac

if [ "$MODE" = "--backend" ]; then
  echo "${backend}"
else
  echo "gpu=${gpu:-unknown}"
  echo "driver=${driver:-unknown}"
  echo "cuda=${cuda:-none}"
  echo "backend=${backend}"
fi
