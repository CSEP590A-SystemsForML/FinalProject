#!/bin/bash

set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat << EOF
Usage:
  ./run.sh

Environment Variables:
  MODEL          Hugging Face model name
                 Default: ibm-granite/granite-4.1-3b

  DTYPE          Quantization mode (drives the quantized_local_lm optimization)
                 Values: bf16, fp8
                 Default: bf16
                 Map: quantized_local_lm=off -> DTYPE=bf16; on -> DTYPE=fp8

  QUANTIZE_KV_CACHE  fp8 KV cache (the quantized_kv_cache optimization)
                 Values: true, false
                 Default: true
                 Set false for a baseline run without fp8 kv-cache.

  MAX_NUM_SEQS   Maximum concurrent sequences
                 Default:
                   bf16 -> 8
                   fp8  -> 16

  HOST           Bind host for local services
                 Default: 127.0.0.1
                 Set HOST=0.0.0.0 only for Colab/demo environments that require it.

  ROUTER_PORT    vLLM OpenAI-compatible router port
                 Default: 7654

  API_PORT       FastAPI resolution server port
                 Default: 8001

  VLLM_BIN       vLLM executable
                 Default: vllm

Examples:
  ./run.sh

  DTYPE=fp8 ./run.sh

  MODEL=ibm-granite/granite-4.1-3b ./run.sh

  DTYPE=fp8 MAX_NUM_SEQS=32 ./run.sh

Fixed Configuration:
  Max Model Length:         8192
  GPU Memory Utilization:   0.95
  Swap Space:               8 GB
  Tensor Parallel Size:     1
  Prefix Caching:           Enabled
  Trust Remote Code:        Enabled
  Request Logging:          Disabled
  KV Cache DType:           FP8 when QUANTIZE_KV_CACHE=true (default), else vLLM default

Service URLs with defaults:
  vLLM router:              http://127.0.0.1:7654/v1
  FastAPI server:           http://127.0.0.1:8001
  FastAPI docs:             http://127.0.0.1:8001/docs
  FastAPI OpenAPI schema:   http://127.0.0.1:8001/openapi.json
  FastAPI health:           http://127.0.0.1:8001/health

MVP note:
  The tool server is not started as a standalone service by this script.
  server/tools.py imports the current tool functions directly.
EOF
    exit 0
fi

MODEL="${MODEL:-ibm-granite/granite-4.1-3b}"
DTYPE="${DTYPE:-bf16}"
HOST="${HOST:-127.0.0.1}"
ROUTER_PORT="${ROUTER_PORT:-7654}"
API_PORT="${API_PORT:-8001}"
VLLM_BIN="${VLLM_BIN:-vllm}"
# quantized_kv_cache optimization: defaults on (current behavior). Set
# QUANTIZE_KV_CACHE=false for a baseline run without fp8 kv-cache.
QUANTIZE_KV_CACHE="${QUANTIZE_KV_CACHE:-true}"

# Dynamic defaults based on dtype.
if [ -z "${MAX_NUM_SEQS:-}" ]; then
    if [ "$DTYPE" = "fp8" ]; then
        MAX_NUM_SEQS=16
    else
        MAX_NUM_SEQS=8
    fi
fi

VLLM_CMD=(
    "$VLLM_BIN" serve "$MODEL"
    --host "$HOST"
    --port "$ROUTER_PORT"
    --download-dir /tmp
    --max-model-len 8192
    --gpu-memory-utilization 0.95
    --tensor-parallel-size 1
    --enable-prefix-caching
    --max-num-seqs "$MAX_NUM_SEQS"
    --trust-remote-code
    --no-enable-log-requests
)

# quantized_kv_cache: only pass fp8 kv-cache when enabled, so the baseline can
# run without it for a fair comparison.
if [ "$QUANTIZE_KV_CACHE" = "true" ]; then
    VLLM_CMD+=(--kv-cache-dtype fp8)
fi

if [ "$DTYPE" = "fp8" ]; then
    VLLM_CMD+=(
        --dtype bfloat16
        --quantization fp8
    )
else
    VLLM_CMD+=(
        --dtype bfloat16
    )
fi

"${VLLM_CMD[@]}" &
VLLM_PID=$!

python3.12 -m uvicorn server.server:app \
    --host "$HOST" \
    --port "$API_PORT" &
FASTAPI_PID=$!

cleanup() {
    kill "$VLLM_PID" "$FASTAPI_PID" 2>/dev/null || true
}

trap cleanup EXIT SIGINT SIGTERM

echo "Started services:"
echo "  vLLM router : http://$HOST:$ROUTER_PORT/v1"
echo "  FastAPI     : http://$HOST:$API_PORT"
echo "  API docs    : http://$HOST:$API_PORT/docs"
echo "  OpenAPI     : http://$HOST:$API_PORT/openapi.json"
echo "  Health      : http://$HOST:$API_PORT/health"
echo ""
echo "Run benchmark with:"
echo "  ROUTER_BASE_URL=http://$HOST:$ROUTER_PORT/v1 python3.12 local-inference/main.py --server-url http://$HOST:$API_PORT"

wait