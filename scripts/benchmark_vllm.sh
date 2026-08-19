#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the model ID or local model path}"
: "${NUM_PROMPTS:?Set NUM_PROMPTS to the request count used by the experiment}"

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_PATH}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
SEED="${SEED:-0}"
RESULT_DIR="${RESULT_DIR:-results/raw}"
RESULT_FILE="${RESULT_FILE:-$RESULT_DIR/vllm_c${MAX_CONCURRENCY}.log}"

mkdir -p "$RESULT_DIR"

echo "Benchmarking vLLM endpoint: $BASE_URL"
echo "model=$SERVED_MODEL_NAME prompts=$NUM_PROMPTS concurrency=$MAX_CONCURRENCY input=$INPUT_LEN output=$OUTPUT_LEN"

vllm bench serve \
  --backend vllm \
  --base-url "$BASE_URL" \
  --model "$SERVED_MODEL_NAME" \
  --dataset-name random \
  --num-prompts "$NUM_PROMPTS" \
  --random-input-len "$INPUT_LEN" \
  --random-output-len "$OUTPUT_LEN" \
  --random-range-ratio 0 \
  --request-rate "$REQUEST_RATE" \
  --max-concurrency "$MAX_CONCURRENCY" \
  --seed "$SEED" \
  2>&1 | tee "$RESULT_FILE"
