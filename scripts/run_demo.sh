#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

if ! command -v cmake >/dev/null 2>&1; then
  echo "[ERROR] cmake not found. Install with: sudo apt update && sudo apt install -y cmake"
  exit 1
fi

if ! command -v g++ >/dev/null 2>&1 && ! command -v clang++ >/dev/null 2>&1; then
  echo "[ERROR] No C++ compiler found. Install with: sudo apt update && sudo apt install -y build-essential"
  exit 1
fi

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" --config Release

BIN="${BUILD_DIR}/darknet_xitao_dgemm_demo"
if [[ ! -x "${BIN}" ]]; then
  BIN="${BUILD_DIR}/Release/darknet_xitao_dgemm_demo"
fi

if [[ $# -eq 0 ]]; then
  if command -v nproc >/dev/null 2>&1; then
    THREADS="$(nproc)"
  else
    THREADS="8"
  fi

  echo "[INFO] No CLI args provided. Running default CNN demo (OpenMP vs XiTAO)."
  "${BIN}" \
    --backend both \
    --channels 64 \
    --height 56 \
    --width 56 \
    --kernel 3 \
    --stride 1 \
    --pad 1 \
    --filters 64 \
    --block-rows 64 \
    --threads "${THREADS}" \
    --warmup 2 \
    --iters 10
else
  "${BIN}" "$@"
fi
