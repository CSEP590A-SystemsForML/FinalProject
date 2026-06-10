#!/usr/bin/env bash
# tpu_setup.sh — one-shot setup of the Cost-Optimizing LLM Router on a TPU VM.
#
# What this does (idempotent where possible):
#   1. Verifies Python 3.12 is on PATH.
#   2. Installs base + colab (vllm-tpu) requirements.
#   3. Patches run.sh to drop CLI flags that vllm-tpu does not accept
#      (--swap-space, --disable-log-requests).
#   4. chmod +x install.sh run.sh
#   5. Recreates the metrics DB (older schemas miss the `label` column) and
#      registers a baseline run row.
#   6. Runs the no-GPU end-to-end smoke test.
#   7. Optionally launches run.sh in the background and waits for the vLLM
#      router to become ready (skip with NO_LAUNCH=1).
#
# Usage:
#   ./scripts/tpu_setup.sh                 # full setup + launch services
#   NO_LAUNCH=1 ./scripts/tpu_setup.sh     # setup only, do not start run.sh
#   SKIP_INSTALL=1 ./scripts/tpu_setup.sh  # skip pip install step
#
# Env knobs forwarded to run.sh:
#   MODEL, DTYPE, QUANTIZE_KV_CACHE, MAX_NUM_SEQS, HOST, ROUTER_PORT, API_PORT
#
# Requires: server/.env already contains API_TOKEN (or export it in your shell).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
RUN_ID="${RUN_ID:-baseline_001}"
QUANTIZE_KV_CACHE="${QUANTIZE_KV_CACHE:-false}"
ROUTER_PORT="${ROUTER_PORT:-7654}"
API_PORT="${API_PORT:-8001}"
HOST="${HOST:-127.0.0.1}"
RUN_LOG="${RUN_LOG:-/tmp/run.log}"
READY_TIMEOUT_SECS="${READY_TIMEOUT_SECS:-1800}"   # 30 min cap for first weight download + compile

log() { printf '\n=== %s ===\n' "$*"; }

# ---------------------------------------------------------------------------
# 1. Python 3.12 check
# ---------------------------------------------------------------------------
log "1/7 Verifying Python 3.12"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: $PYTHON_BIN not found on PATH" >&2
    exit 1
fi
"$PYTHON_BIN" -c "import sys; assert sys.version_info[:2] == (3, 12), sys.version"
"$PYTHON_BIN" --version

# ---------------------------------------------------------------------------
# 2. Install deps (install.sh requires PYTHON env var when the default
#    `python` is not 3.12)
# ---------------------------------------------------------------------------
if [[ "${SKIP_INSTALL:-0}" != "1" ]]; then
    log "2/7 Installing requirements (base + colab/vllm-tpu)"
    chmod +x install.sh
    PYTHON="$PYTHON_BIN" ./install.sh colab
else
    log "2/7 Skipping install (SKIP_INSTALL=1)"
fi

# ---------------------------------------------------------------------------
# 3. Patch run.sh: vllm-tpu rejects --swap-space and --disable-log-requests
# ---------------------------------------------------------------------------
log "3/7 Patching run.sh for vllm-tpu CLI compatibility"
if grep -q -- '--swap-space 8' run.sh; then
    sed -i '/--swap-space 8/d' run.sh
    echo "  removed --swap-space 8"
fi
if grep -q -- '--disable-log-requests' run.sh; then
    sed -i 's/--disable-log-requests/--no-enable-log-requests/' run.sh
    echo "  renamed --disable-log-requests -> --no-enable-log-requests"
fi

# ---------------------------------------------------------------------------
# 4. Make launch scripts executable
# ---------------------------------------------------------------------------
log "4/7 chmod +x install.sh run.sh"
chmod +x install.sh run.sh

# ---------------------------------------------------------------------------
# 5. Reset metrics DB if it lacks the `label` column, then register the run
# ---------------------------------------------------------------------------
log "5/7 Initializing metrics DB and registering run_id=$RUN_ID (baseline)"
"$PYTHON_BIN" - <<'PY'
import sqlite3, pathlib
db = pathlib.Path("server/metrics/metrics.db")
if db.exists():
    cols = [r[1] for r in sqlite3.connect(db).execute("PRAGMA table_info(optimizations)")]
    if "label" not in cols:
        print(f"  stale schema in {db} (no `label` col) -> deleting")
        db.unlink()
PY
"$PYTHON_BIN" server/metrics/create_run.py --run-id "$RUN_ID" --label Baseline --baseline

# ---------------------------------------------------------------------------
# 6. Smoke test (no GPU, no API key needed)
# ---------------------------------------------------------------------------
log "6/7 Running e2e smoke test (math, 2 problems)"
"$PYTHON_BIN" scripts/e2e_smoke.py --domain math --per-domain 2

# ---------------------------------------------------------------------------
# 7. Launch vLLM router + FastAPI server and wait for readiness
# ---------------------------------------------------------------------------
if [[ "${NO_LAUNCH:-0}" == "1" ]]; then
    log "7/7 NO_LAUNCH=1 set — skipping run.sh launch"
    echo "When ready, start services manually with:"
    echo "  QUANTIZE_KV_CACHE=$QUANTIZE_KV_CACHE ./run.sh"
    exit 0
fi

log "7/7 Launching ./run.sh (logs -> $RUN_LOG)"

# Clean up any stale listeners on the target ports
if ss -tln 2>/dev/null | grep -qE ":($ROUTER_PORT|$API_PORT) "; then
    echo "  stale listener on $ROUTER_PORT/$API_PORT — killing vllm/uvicorn"
    pkill -f "vllm serve" 2>/dev/null || true
    pkill -f "uvicorn server.server:app" 2>/dev/null || true
    sleep 2
fi

: >"$RUN_LOG"
QUANTIZE_KV_CACHE="$QUANTIZE_KV_CACHE" \
HOST="$HOST" ROUTER_PORT="$ROUTER_PORT" API_PORT="$API_PORT" \
nohup ./run.sh >"$RUN_LOG" 2>&1 &
RUN_PID=$!
disown || true
echo "  run.sh pid=$RUN_PID"

log "Waiting up to ${READY_TIMEOUT_SECS}s for vLLM /v1/models"
deadline=$(( $(date +%s) + READY_TIMEOUT_SECS ))
while :; do
    if curl -sf "http://${HOST}:${ROUTER_PORT}/v1/models" -o /dev/null 2>&1; then
        echo "  vLLM router READY at http://${HOST}:${ROUTER_PORT}/v1"
        break
    fi
    if ! kill -0 "$RUN_PID" 2>/dev/null; then
        echo "ERROR: run.sh exited before vLLM became ready — tail of $RUN_LOG:" >&2
        tail -80 "$RUN_LOG" >&2
        exit 1
    fi
    if (( $(date +%s) >= deadline )); then
        echo "ERROR: vLLM not ready within ${READY_TIMEOUT_SECS}s — tail of $RUN_LOG:" >&2
        tail -80 "$RUN_LOG" >&2
        exit 1
    fi
    sleep 10
done

cat <<EOF

Services up:
  vLLM router  : http://${HOST}:${ROUTER_PORT}/v1
  FastAPI      : http://${HOST}:${API_PORT}
  API docs     : http://${HOST}:${API_PORT}/docs
  Health       : http://${HOST}:${API_PORT}/health
  run.sh logs  : $RUN_LOG  (pid=$RUN_PID)

Drive the benchmark in another terminal:
  $PYTHON_BIN local-inference/main.py --run-id $RUN_ID \\
      --server-url http://${HOST}:${API_PORT} --limit 3

Then analyze:
  $PYTHON_BIN server/metrics/analysis_script.py
EOF
