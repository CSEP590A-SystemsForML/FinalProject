#!/bin/bash
# One-shot: take a freshly provisioned VM from zero to a full A/B analysis.
#
# Phases:
#   1. install      deps (requirements/base.txt + platform extras), into a venv
#   2. serve        start the vLLM router + FastAPI resolution server (run.sh)
#   3. health       wait until both the router and the server are responding
#   4. benchmark    run the SAME problem set through BOTH routing strategies
#                   (difficulty/legacy and confidence/ladder) -> metrics DB
#   5. analyze      emit comparison tables + cost/frontier plots
#
# Designed to be safe to re-run and to degrade gracefully:
#   - No API key (or MODE=smoke) -> runs the deterministic e2e smoke instead of
#     live model calls, so you still get a populated DB + analysis.
#   - SERVE lets you reuse an already-running router/server (e.g. a shared TPU).
#
# Usage (from anywhere; the script cd's to the repo root):
#   API_TOKEN=sk-or-... scripts/bootstrap_vm.sh
#
# Common overrides (env vars):
#   MODE=live|smoke         default live if an API key is set, else smoke
#   SERVE=all|server|none   what to launch. all=router+server (default),
#                           server=only FastAPI (reuse external ROUTER_BASE_URL),
#                           none=assume both already running.
#   INSTALL_TARGET=colab|mac|none    default: auto (mac on Darwin, else colab)
#   PER_DOMAIN=8            problems per domain per strategy
#   ID_MIN= / ID_MAX=      optional inclusive problem_id range to trim a big set
#                          for a demo (e.g. ID_MAX=1150). Empty = no range filter.
#   DOMAINS=math,reasoning,factual,code
#   RUN_TAG=ab              run_id prefix -> <tag>_difficulty / <tag>_confidence
#   HOST / ROUTER_PORT / API_PORT / ROUTER_BASE_URL / DTYPE / ACCEL  (see run.sh)
#   PYTHON=python3.12      interpreter used to build the venv
set -euo pipefail

# --- locate repo root (this script lives in <root>/scripts) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

log() { printf '\n\033[1m[bootstrap]\033[0m %s\n' "$*"; }
die() { printf '\n\033[31m[bootstrap] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# --- config ---
PYTHON="${PYTHON:-python3.12}"
VENV="${VENV:-.venv}"
SERVE="${SERVE:-all}"
HOST="${HOST:-127.0.0.1}"
ROUTER_PORT="${ROUTER_PORT:-7654}"
API_PORT="${API_PORT:-8001}"
ROUTER_BASE_URL="${ROUTER_BASE_URL:-http://${HOST}:${ROUTER_PORT}/v1}"
SERVER_URL="http://${HOST}:${API_PORT}"
PER_DOMAIN="${PER_DOMAIN:-8}"
DOMAINS="${DOMAINS:-math,reasoning,factual,code}"
RUN_TAG="${RUN_TAG:-ab}"
HEALTH_TIMEOUT="${HEALTH_TIMEOUT:-1200}"   # vLLM weight load can be slow on first boot
LOG_DIR="${LOG_DIR:-/tmp/bootstrap_vm}"
mkdir -p "$LOG_DIR"

API_KEY="${API_TOKEN:-${OPENROUTER_API_KEY:-${LITELLM_API_KEY:-}}}"
MODE="${MODE:-$([ -n "$API_KEY" ] && echo live || echo smoke)}"

if [ "${INSTALL_TARGET:-auto}" = "auto" ]; then
    [ "$(uname -s)" = "Darwin" ] && INSTALL_TARGET=mac || INSTALL_TARGET=colab
fi

STARTED_SERVICES_PID=""
cleanup() {
    if [ -n "$STARTED_SERVICES_PID" ]; then
        log "stopping services we started (pgid $STARTED_SERVICES_PID)"
        kill -- "-$STARTED_SERVICES_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

log "repo=$REPO_ROOT mode=$MODE serve=$SERVE install=$INSTALL_TARGET"
log "router=$ROUTER_BASE_URL server=$SERVER_URL domains=$DOMAINS per_domain=$PER_DOMAIN tag=$RUN_TAG"

# ============================================================
# Phase 1: install
# ============================================================
if [ "${SKIP_INSTALL:-0}" = "1" ] || [ "$INSTALL_TARGET" = "none" ]; then
    log "phase 1/5 install: skipped"
else
    log "phase 1/5 install: building venv at $VENV ($PYTHON) + $INSTALL_TARGET deps"
    command -v "$PYTHON" >/dev/null 2>&1 || die "$PYTHON not found (need Python 3.12)."
    "$PYTHON" -c "import sys; assert sys.version_info[:2]==(3,12)" \
        || die "Python 3.12 required (got $($PYTHON -V 2>&1))."
    [ -d "$VENV" ] || "$PYTHON" -m venv "$VENV"
    VPY="$VENV/bin/python"
    "$VPY" -m pip install --quiet --upgrade pip setuptools wheel
    "$VPY" -m pip install --quiet -r requirements/base.txt
    "$VPY" -m pip install --quiet -r "requirements/${INSTALL_TARGET}.txt"
    "$VPY" -m pip install --quiet matplotlib >/dev/null 2>&1 || true   # for analysis plots
fi
# Prefer the venv interpreter for everything downstream when present.
PY="${PY:-$([ -x "$VENV/bin/python" ] && echo "$VENV/bin/python" || echo "$PYTHON")}"
export PYTHON="$PY"
log "using interpreter: $PY"

# ============================================================
# Phase 2: serve
# ============================================================
wait_http() {  # url, timeout_s, label
    local url="$1" timeout="$2" label="$3" waited=0
    log "waiting for $label ($url) up to ${timeout}s"
    until curl -fsS -m 3 "$url" >/dev/null 2>&1; do
        sleep 3; waited=$((waited+3))
        [ "$waited" -ge "$timeout" ] && die "$label did not become healthy within ${timeout}s (see $LOG_DIR)."
    done
    log "$label is up (${waited}s)"
}

# The smoke harness (scripts/e2e_smoke.py) runs its OWN in-process server and
# mocks the model calls, so it needs neither external services nor a router.
if [ "$MODE" = "smoke" ]; then
    log "phase 2/5 serve: skipped (MODE=smoke runs an in-process server)"
elif [ "$SERVE" = "all" ]; then
    log "phase 2/5 serve: launching vLLM router + FastAPI server via run.sh"
    # New process group so cleanup() can take the whole tree down.
    HOST="$HOST" ROUTER_PORT="$ROUTER_PORT" API_PORT="$API_PORT" \
        setsid bash run.sh >"$LOG_DIR/services.log" 2>&1 &
    STARTED_SERVICES_PID=$!
elif [ "$SERVE" = "server" ]; then
    log "phase 2/5 serve: launching ONLY the FastAPI server (reusing router at $ROUTER_BASE_URL)"
    [ -n "$API_KEY" ] && export API_TOKEN="$API_KEY"
    setsid "$PY" -m uvicorn server.server:app --host "$HOST" --port "$API_PORT" \
        >"$LOG_DIR/server.log" 2>&1 &
    STARTED_SERVICES_PID=$!
elif [ "$SERVE" = "none" ]; then
    log "phase 2/5 serve: skipped (assuming router + server already running)"
else
    die "unknown SERVE=$SERVE (use all|server|none)"
fi

# ============================================================
# Phase 3: health
# ============================================================
if [ "$MODE" = "smoke" ]; then
    log "phase 3/5 health: skipped (in-process smoke server)"
else
    log "phase 3/5 health"
    wait_http "$SERVER_URL/health" "$HEALTH_TIMEOUT" "FastAPI server"
    if [ "${CHECK_ROUTER:-1}" = "1" ]; then
        wait_http "${ROUTER_BASE_URL%/}/models" "$HEALTH_TIMEOUT" "vLLM router" || true
    fi
fi

# ============================================================
# Phase 4: benchmark
# ============================================================
if [ "$MODE" = "smoke" ]; then
    log "phase 4/5 benchmark: MODE=smoke -> deterministic e2e (no API key needed)"
    "$PY" scripts/e2e_smoke.py --per-domain "$PER_DOMAIN" || die "e2e smoke failed"
else
    log "phase 4/5 benchmark: live A/B over both routing strategies"
    [ -n "$API_KEY" ] && export API_TOKEN="$API_KEY"
    for strat in difficulty confidence; do
        "$PY" server/metrics/create_run.py --run-id "${RUN_TAG}_${strat}" \
            --label "${RUN_TAG} ${strat}" --quiet || true
    done
    ROUTER_BASE_URL="$ROUTER_BASE_URL" PYTHON="$PY" \
        bash scripts/run_strategy_ab.sh "$SERVER_URL" "$PER_DOMAIN" "$DOMAINS" "$RUN_TAG" \
        2>&1 | tee "$LOG_DIR/benchmark.log"
fi

# ============================================================
# Phase 5: analyze
# ============================================================
log "phase 5/5 analyze: tables + plots"
"$PY" server/metrics/analysis_script.py | tee "$LOG_DIR/analysis.log"

log "DONE."
log "  metrics DB : server/metrics/metrics.db"
log "  tables/plots: server/metrics/outputs/"
log "  logs        : $LOG_DIR/"
if [ "$MODE" = "live" ]; then
    log "  compare runs: ${RUN_TAG}_difficulty  vs  ${RUN_TAG}_confidence  (see run_summary.csv)"
fi
