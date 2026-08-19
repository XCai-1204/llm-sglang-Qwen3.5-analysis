#!/usr/bin/env bash
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the model ID or local model path}"
: "${NUM_PROMPTS:?Set NUM_PROMPTS to the request count used by the experiment}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$MODEL_PATH}"
MAX_CONCURRENCY="${MAX_CONCURRENCY:-1}"
INPUT_LEN="${INPUT_LEN:-512}"
OUTPUT_LEN="${OUTPUT_LEN:-128}"
REQUEST_RATE="${REQUEST_RATE:-inf}"
RESULT_DIR="${RESULT_DIR:-results/raw}"
RESULT_FILE="${RESULT_FILE:-$RESULT_DIR/sglang_c${MAX_CONCURRENCY}.jsonl}"
LOG_FILE="${LOG_FILE:-$RESULT_DIR/sglang_c${MAX_CONCURRENCY}.log}"

mkdir -p "$RESULT_DIR"

echo "Benchmarking SGLang endpoint: http://$HOST:$PORT"
echo "model=$SERVED_MODEL_NAME prompts=$NUM_PROMPTS concurrency=$MAX_CONCURRENCY input=$INPUT_LEN output=$OUTPUT_LEN"

python3 -m sglang.bench_serving \
  --backend sglang \
  --host "$HOST" \
  --port "$PORT" \
  --model "$SERVED_MODEL_NAME" \
  --dataset-name random \
  --num-prompts "$NUM_PROMPTS" \
  --random-input-len "$INPUT_LEN" \
  --random-output-len "$OUTPUT_LEN" \
  --random-range-ratio 0 \
  --request-rate "$REQUEST_RATE" \
  --max-concurrency "$MAX_CONCURRENCY" \
  --output-file "$RESULT_FILE" \
  --output-details \
  2>&1 | tee "$LOG_FILE"
