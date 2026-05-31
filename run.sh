#!/bin/bash

set -e

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    cat << EOF
Usage:
  ./serve.sh

Environment Variables:
  MODEL          Hugging Face model name
                 Default: Qwen/Qwen3.5-4B

  DTYPE          Quantization mode
                 Values: bf16, fp8
                 Default: bf16

  MAX_NUM_SEQS   Maximum concurrent sequences
                 Default:
                   bf16 -> 8
                   fp8  -> 16

Examples:
  ./serve.sh

  ./serve.sh DTYPE=fp8

  ./serve.sh MODEL=Qwen/Qwen3.5-8B

  ./serve.sh DTYPE=fp8 MAX_NUM_SEQS=32

Fixed Configuration:
  Port:                     8000
  Max Model Length:         8192
  GPU Memory Utilization:   0.95
  Swap Space:               8 GB
  Tensor Parallel Size:     1
  Prefix Caching:           Enabled
  Trust Remote Code:        Enabled
  Request Logging:          Disabled
  KV Cache DType:           FP8
EOF
    exit 0
fi

MODEL="${MODEL:-Qwen/Qwen3.5-4B}"

# Supported values: bf16, fp8
DTYPE="${DTYPE:-bf16}"

# Dynamic defaults based on dtype
if [ -z "${MAX_NUM_SEQS:-}" ]; then
    if [ "$DTYPE" = "fp8" ]; then
        MAX_NUM_SEQS=16
    else
        MAX_NUM_SEQS=8
    fi
fi

CMD=(
    ~/.local/bin/vllm serve "$MODEL"
    --host 0.0.0.0
    --port 8000
    --download-dir /tmp
    --max-model-len 8192
    --gpu-memory-utilization 0.95
    --swap-space 8
    --tensor-parallel-size 1
    --enable-prefix-caching
    --max-num-seqs "$MAX_NUM_SEQS"
    --trust-remote-code
    --disable-log-requests
    --kv-cache-dtype fp8
    --reasoning-parser qwen3
)

if [ "$DTYPE" = "fp8" ]; then
    CMD+=(
        --dtype bfloat16
        --quantization fp8
    )
else
    CMD+=(
        --dtype bfloat16
    )
fi

"${CMD[@]}" &
VLLM_PID=$!

# Start FastAPI
python3.12 -m uvicorn server.server:app \
    --host 0.0.0.0 \
    --port 8001 &
FASTAPI_PID=$!

# Start FastMCP
python3.12 -m uvicorn tool-sever.server:mcp \
    --host 0.0.0.0 \
    --port 8002 &
FASTMCP_PID=$!

echo "Started services:"
echo "  vLLM     : http://localhost:8000"
echo "  FastAPI  : http://localhost:8001"
echo "  FastMCP  : http://localhost:8002"

# Cleanup on Ctrl+C
trap 'kill $VLLM_PID $FASTAPI_PID $FASTMCP_PID' SIGINT SIGTERM

wait