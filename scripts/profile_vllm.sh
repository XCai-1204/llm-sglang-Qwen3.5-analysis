#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the model ID or local model path}"

NSYS_BIN="${NSYS_BIN:-nsys}"
TP_SIZE="${TP_SIZE:-2}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
PROFILE_OUTPUT="${PROFILE_OUTPUT:-profiles/vllm_qwen35}"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"

export CUDA_VISIBLE_DEVICES
mkdir -p "$(dirname "$PROFILE_OUTPUT")"

# Run this script in terminal A. After the server is ready, run the benchmark
# from terminal B, then stop this process with Ctrl+C so Nsight writes the report.
read -r -a extra_args <<< "$EXTRA_SERVER_ARGS"

exec "$NSYS_BIN" profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --cuda-graph-trace=graph \
  --force-overwrite=true \
  --output="$PROFILE_OUTPUT" \
  vllm serve "$MODEL_PATH" \
  --tensor-parallel-size "$TP_SIZE" \
  --host "$HOST" \
  --port "$PORT" \
  "${extra_args[@]}"
