#!/bin/bash

set -e

if [[ "$1" == "--help" || "$1" == "-h" ]]; then
    cat << EOF
Usage:
  ./run.sh

Environment Variables:
  MODEL          HF model name (must be a JAX-native arch on TPU).
                 Default: Qwen/Qwen3-4B
  DTYPE          bf16 | fp8   (fp8 is GPU-only; ignored on TPU)
                 Default: bf16
  MAX_NUM_SEQS   Max concurrent sequences.
                 Default: 8

Fixed Configuration:
  Port:                     8000
  Max Model Length:         8192
  Tensor Parallel Size:     1
  Prefix Caching:           Enabled
  Trust Remote Code:        Enabled
EOF
    exit 0
fi

MODEL="${MODEL:-Qwen/Qwen3-4B}"
DTYPE="${DTYPE:-bf16}"

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
    --tensor-parallel-size 1
    --enable-prefix-caching
    --max-num-seqs "$MAX_NUM_SEQS"
    --trust-remote-code
    --reasoning-parser qwen3
    --dtype bfloat16
)
[ "$DTYPE" = "fp8" ] && CMD+=(--quantization fp8)

# Run vLLM in its own process group so we can kill all descendants at once.
setsid "${CMD[@]}" &
VLLM_PID=$!

python3.12 -m uvicorn server.server:app --host 0.0.0.0 --port 8001 &
FASTAPI_PID=$!

python3.12 -m uvicorn tool-server.server:mcp --host 0.0.0.0 --port 8002 &
FASTMCP_PID=$!

echo "Started services:"
echo "  vLLM     : http://localhost:8000 (pgid=$VLLM_PID)"
echo "  FastAPI  : http://localhost:8001"
echo "  FastMCP  : http://localhost:8002"

cleanup() {
    echo
    echo "Shutting down (releasing TPU)..."
    # Kill vLLM's whole process group (EngineCore + APIServer children)
    kill -TERM -"$VLLM_PID" 2>/dev/null || true
    kill -TERM "$FASTAPI_PID" "$FASTMCP_PID" 2>/dev/null || true
    # Give EngineCore a moment to release libtpu.so, then SIGKILL stragglers
    sleep 3
    kill -KILL -"$VLLM_PID" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

wait