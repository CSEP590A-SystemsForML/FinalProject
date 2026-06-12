#!/bin/bash
# A/B benchmark: run the SAME problem subset through both routing strategies so
# their metrics (solve rate, cost, escalation, rung usage) are directly comparable.
#
# Usage:
#   ROUTER_BASE_URL=http://127.0.0.1:7654/v1 \
#   scripts/run_strategy_ab.sh [SERVER_URL] [PER_DOMAIN] [DOMAINS_CSV] [RUN_TAG]
#
# Examples:
#   scripts/run_strategy_ab.sh http://127.0.0.1:8001 8 "math,reasoning,factual,code" ab1
#
# Produces two runs in the metrics DB:
#   <RUN_TAG>_difficulty   (legacy: router pick + single escalation to strongest)
#   <RUN_TAG>_confidence   (ladder: confidence-routed start rung + gradual climb)
set -euo pipefail

cd "$(dirname "$0")/.."

SERVER="${1:-http://127.0.0.1:8001}"
PER_DOMAIN="${2:-8}"
DOMAINS="${3:-math,reasoning,factual,code}"
RUN_TAG="${4:-ab}"
PY="${PYTHON:-.venv/bin/python}"

# Optional inclusive problem_id range to trim a large set for a demo
# (e.g. ID_MIN=1000 ID_MAX=1150). Empty = no range filter.
ID_MIN="${ID_MIN:-}"
ID_MAX="${ID_MAX:-}"
RANGE_ARGS=()
[ -n "$ID_MIN" ] && RANGE_ARGS+=(--id-min "$ID_MIN")
[ -n "$ID_MAX" ] && RANGE_ARGS+=(--id-max "$ID_MAX")

run_strategy() {
    local strategy="$1" run_id="$2"
    echo "##### STRATEGY=${strategy} RUN_ID=${run_id} #####"
    IFS=',' read -ra DOMS <<< "$DOMAINS"
    for dom in "${DOMS[@]}"; do
        echo "### domain=${dom} ###"
        "$PY" local-inference/main.py \
            --run-id "$run_id" \
            --server-url "$SERVER" \
            --strategy "$strategy" \
            --domain "$dom" \
            --limit "$PER_DOMAIN" \
            --max-active 2 \
            ${RANGE_ARGS[@]+"${RANGE_ARGS[@]}"}
    done
}

run_strategy difficulty "${RUN_TAG}_difficulty"
run_strategy confidence "${RUN_TAG}_confidence"
echo "##### STRATEGY AB DONE #####"
