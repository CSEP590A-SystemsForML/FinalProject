#!/bin/bash
# Throughput-vs-quantization sweep for the vLLM router model.
#
# For each quantization config it: (re)starts ONLY the vLLM router via
# `LAUNCH=router ./run.sh`, waits for the model to load, runs
# scripts/profile_vllm.py against it, then tears the router down. Finally it
# prints a side-by-side comparison table and writes summary.json.
#
# This answers the "profile throughput" / "optimization opportunities" part of
# the brief: how much does fp8 (or TPU tpu_int8) weight quant + fp8 kv-cache buy
# us in tokens/sec on this hardware?
#
# Configs are "DTYPE:QUANTIZE_KV_CACHE" pairs. On TPU, run.sh maps fp8 weight
# quant -> tpu_int8 automatically (v5e has no fp8 weights).
#
# Usage:
#   scripts/profile_quant_sweep.sh
#   CONFIGS="bf16:false bf16:true fp8:true" NUM_REQUESTS=128 CONCURRENCY=16 \
#       scripts/profile_quant_sweep.sh
#
# Env overrides:
#   CONFIGS        space-separated DTYPE:KV pairs   (default: bf16:false bf16:true fp8:true)
#   NUM_REQUESTS   measured requests per config     (default: 64)
#   CONCURRENCY    in-flight requests               (default: 8)
#   MAX_TOKENS     output tokens per request        (default: 256)
#   PROMPT_TOKENS  approx prompt length             (default: 512)
#   OUT_DIR        where JSON reports land          (default: /tmp/vllm_profile)
#   HOST/ROUTER_PORT/MODEL/ACCEL/VLLM_BIN          passed through to run.sh
#   HEALTH_TIMEOUT seconds to wait for model load   (default: 1200)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

log() { printf '\n\033[1m[sweep]\033[0m %s\n' "$*"; }
die() { printf '\n\033[31m[sweep] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

CONFIGS="${CONFIGS:-bf16:false bf16:true fp8:true}"
NUM_REQUESTS="${NUM_REQUESTS:-64}"
CONCURRENCY="${CONCURRENCY:-8}"
MAX_TOKENS="${MAX_TOKENS:-256}"
PROMPT_TOKENS="${PROMPT_TOKENS:-512}"
OUT_DIR="${OUT_DIR:-/tmp/vllm_profile}"
HOST="${HOST:-127.0.0.1}"
ROUTER_PORT="${ROUTER_PORT:-7654}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1200}"
PY="${PYTHON:-python3.12}"
ROUTER_BASE_URL="http://${HOST}:${ROUTER_PORT}/v1"
ROUTER_HEALTH="http://${HOST}:${ROUTER_PORT}/health"

mkdir -p "$OUT_DIR"
command -v "$PY" >/dev/null 2>&1 || PY="python3"

kill_router() {
    # Targeted teardown: free the router port without touching SSH or other procs.
    if command -v fuser >/dev/null 2>&1; then
        fuser -k "${ROUTER_PORT}/tcp" >/dev/null 2>&1 || true
    elif command -v lsof >/dev/null 2>&1; then
        lsof -ti "tcp:${ROUTER_PORT}" 2>/dev/null | xargs -r kill 2>/dev/null || true
    fi
    sleep 3
}

wait_for_router() {
    local deadline=$(( $(date +%s) + HEALTH_TIMEOUT ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        if curl -fsS -m 2 "$ROUTER_HEALTH" >/dev/null 2>&1; then
            return 0
        fi
        sleep 5
    done
    return 1
}

profile_one() {
    local dtype="$1" kv="$2"
    local tag="${dtype}_kv-${kv}"
    local out_json="${OUT_DIR}/${tag}.json"
    local serve_log="${OUT_DIR}/${tag}.serve.log"

    log "config dtype=${dtype} kv_cache_fp8=${kv}  ->  ${out_json}"
    kill_router

    LAUNCH=router DTYPE="$dtype" QUANTIZE_KV_CACHE="$kv" \
        HOST="$HOST" ROUTER_PORT="$ROUTER_PORT" \
        bash run.sh >"$serve_log" 2>&1 &
    local run_pid=$!

    if ! wait_for_router; then
        kill_router
        printf '\033[31m[sweep] router did not become healthy for %s; see %s\033[0m\n' "$tag" "$serve_log" >&2
        return 1
    fi

    PROFILE_LABEL="$tag" DTYPE="$dtype" QUANTIZE_KV_CACHE="$kv" ACCEL="${ACCEL:-auto}" \
        MAX_NUM_SEQS="${MAX_NUM_SEQS:-}" ROUTER_BASE_URL="$ROUTER_BASE_URL" \
        "$PY" scripts/profile_vllm.py \
            --base-url "$ROUTER_BASE_URL" \
            --num-requests "$NUM_REQUESTS" \
            --concurrency "$CONCURRENCY" \
            --max-tokens "$MAX_TOKENS" \
            --prompt-tokens "$PROMPT_TOKENS" \
            --out "$out_json" || printf '\033[31m[sweep] profile failed for %s\033[0m\n' "$tag" >&2

    kill_router
    wait "$run_pid" 2>/dev/null || true
}

IFS=' ' read -ra PAIRS <<< "$CONFIGS"
for pair in "${PAIRS[@]}"; do
    dtype="${pair%%:*}"
    kv="${pair##*:}"
    profile_one "$dtype" "$kv" || true
done

log "aggregating reports in ${OUT_DIR}"
"$PY" - "$OUT_DIR" <<'PY'
import json, sys
from pathlib import Path

out_dir = Path(sys.argv[1])
reports = []
for p in sorted(out_dir.glob("*.json")):
    if p.name == "summary.json":
        continue
    try:
        reports.append(json.loads(p.read_text()))
    except Exception as e:
        print(f"  skip {p.name}: {e}")

if not reports:
    print("  no reports to aggregate")
    raise SystemExit(0)

baseline = None
rows = []
for r in reports:
    res = r["results"]
    cfg = r["config"]
    rows.append({
        "label": cfg.get("label") or f"{cfg.get('dtype')}/{cfg.get('kv_cache')}",
        "dtype": cfg.get("dtype"),
        "kv_cache_fp8": cfg.get("kv_cache"),
        "accel": cfg.get("accel"),
        "out_tok_s": res["output_throughput_tok_s"],
        "req_s": res["request_throughput_req_s"],
        "lat_p50": res["latency_s"]["p50"],
        "lat_p95": res["latency_s"]["p95"],
        "ttft_p50": res["ttft_s"]["p50"],
        "errors": res["errors"],
    })

# Speedup relative to the slowest config (treated as the baseline).
slowest = min(rows, key=lambda x: x["out_tok_s"])["out_tok_s"] or 1.0

hdr = f"{'config':<20}{'out tok/s':>12}{'speedup':>9}{'req/s':>9}{'lat p50':>10}{'lat p95':>10}{'ttft p50':>10}{'err':>5}"
print("\n" + hdr)
print("-" * len(hdr))
for row in sorted(rows, key=lambda x: x["out_tok_s"]):
    print(
        f"{row['label']:<20}"
        f"{row['out_tok_s']:>12.1f}"
        f"{(row['out_tok_s']/slowest):>8.2f}x"
        f"{row['req_s']:>9.2f}"
        f"{row['lat_p50']:>10.3f}"
        f"{row['lat_p95']:>10.3f}"
        f"{row['ttft_p50']:>10.3f}"
        f"{row['errors']:>5}"
    )

summary = {"rows": rows, "speedup_baseline_tok_s": slowest}
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"\nwrote {out_dir / 'summary.json'}")
PY

log "done. JSON reports + summary.json in ${OUT_DIR}"
