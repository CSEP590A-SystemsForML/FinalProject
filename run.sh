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
                 (Supported on CUDA and TPU v5+ / vLLM >= 0.10.)

  ACCEL          Accelerator backend
                 Values: auto, gpu, tpu
                 Default: auto (TPU via /dev/accel0 or GCE accelerator-type
                 metadata, e.g. v5litepod-1; else gpu)
                 On TPU: adds --device tpu; DTYPE=fp8 maps to tpu_int8 weight
                 quant (TPU v5e has no fp8 weights; needs v6e/Ironwood).

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

  LAUNCH         Which services to start
                 Values: all, router
                 Default: all (vLLM router + FastAPI server)
                 router: start only the vLLM router (used by throughput
                 profiling, e.g. scripts/profile_quant_sweep.sh).

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
LAUNCH="${LAUNCH:-all}"
# quantized_kv_cache optimization: defaults on (current behavior). Set
# QUANTIZE_KV_CACHE=false for a baseline run without fp8 kv-cache.
QUANTIZE_KV_CACHE="${QUANTIZE_KV_CACHE:-true}"

# Accelerator backend. Auto-detects a Cloud TPU unless overridden with
# ACCEL=tpu|gpu. On TPU we add `--device tpu` and avoid GPU-only fp8 *weight*
# quantization: TPU v5e has no fp8 weights and uses tpu_int8 instead (fp8
# weights need v6e/Ironwood). fp8 *kv-cache* is fine on TPU v5+ (vLLM >= 0.10).
#
# Detection: older TPUs (v4-) expose /dev/accel0; v5e/v6e expose the chip via
# VFIO, so we also ask the GCE metadata server for the accelerator-type (e.g.
# v5litepod-1, v6e-8). The 1s curl timeout makes this a fast no-op off-GCE.
detect_tpu() {
    [ -e /dev/accel0 ] && return 0
    local acc
    acc="$(curl -s -m 1 -H 'Metadata-Flavor: Google' \
        'http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type' 2>/dev/null || true)"
    case "$acc" in
        v[0-9]*) return 0 ;;
    esac
    return 1
}

ACCEL="${ACCEL:-auto}"
if [ "$ACCEL" = "auto" ]; then
    if detect_tpu; then
        ACCEL="tpu"
    else
        ACCEL="gpu"
    fi
fi

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
    --swap-space 8
    --tensor-parallel-size 1
    --enable-prefix-caching
    --max-num-seqs "$MAX_NUM_SEQS"
    --trust-remote-code
    --disable-log-requests
    --reasoning-parser qwen3
    --dtype bfloat16
)

if [ "$ACCEL" = "tpu" ]; then
    VLLM_CMD+=(--device tpu)
fi

# quantized_kv_cache: only pass fp8 kv-cache when enabled, so the baseline can
# run without it for a fair comparison. (CUDA, and TPU v5+ on vLLM >= 0.10.)
if [ "$QUANTIZE_KV_CACHE" = "true" ]; then
    VLLM_CMD+=(--kv-cache-dtype fp8)
fi

# quantized_local_lm: weight quantization. fp8 on GPU; tpu_int8 on TPU v5e.
if [ "$DTYPE" = "fp8" ]; then
    if [ "$ACCEL" = "tpu" ]; then
        echo "[run.sh] TPU detected: mapping fp8 weight quant -> tpu_int8 (v5e has no fp8 weights)."
        VLLM_CMD+=(--quantization tpu_int8)
    else
        VLLM_CMD+=(--quantization fp8)
    fi
fi

echo "[run.sh] accelerator=$ACCEL dtype=$DTYPE kv_cache_fp8=$QUANTIZE_KV_CACHE model=$MODEL"

"${VLLM_CMD[@]}" &
VLLM_PID=$!

# LAUNCH=router: only the vLLM router (used by throughput profiling). Otherwise
# also start the FastAPI resolution server.
FASTAPI_PID=""
if [ "$LAUNCH" != "router" ]; then
    python3.12 -m uvicorn server.server:app \
        --host "$HOST" \
        --port "$API_PORT" &
    FASTAPI_PID=$!
fi

cleanup() {
    kill "$VLLM_PID" ${FASTAPI_PID:+"$FASTAPI_PID"} 2>/dev/null || true
}

trap cleanup EXIT SIGINT SIGTERM

if [ "$LAUNCH" = "router" ]; then
    echo "Started services (router only):"
    echo "  vLLM router : http://$HOST:$ROUTER_PORT/v1"
    echo ""
    echo "Profile throughput with:"
    echo "  ROUTER_BASE_URL=http://$HOST:$ROUTER_PORT/v1 python3.12 scripts/profile_vllm.py"
else
    echo "Started services:"
    echo "  vLLM router : http://$HOST:$ROUTER_PORT/v1"
    echo "  FastAPI     : http://$HOST:$API_PORT"
    echo "  API docs    : http://$HOST:$API_PORT/docs"
    echo "  OpenAPI     : http://$HOST:$API_PORT/openapi.json"
    echo "  Health      : http://$HOST:$API_PORT/health"
    echo ""
    echo "Run benchmark with:"
    echo "  ROUTER_BASE_URL=http://$HOST:$ROUTER_PORT/v1 python3.12 local-inference/main.py --server-url http://$HOST:$API_PORT"
fi

wait