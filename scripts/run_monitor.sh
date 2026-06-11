#!/usr/bin/env bash
# Append a one-screen progress snapshot to /tmp/run_status.log every INTERVAL seconds.
# Stops once no `local-inference/main.py` process is running for two checks in a row.
set -u

LOG=${LOG:-/tmp/run_status.log}
INTERVAL=${INTERVAL:-30}
DB=${DB:-/home/anabyv/FinalProject/server/metrics/metrics.db}
IDLE_LIMIT=${IDLE_LIMIT:-2}

idle=0
echo "monitor started pid=$$ interval=${INTERVAL}s db=${DB}" > "$LOG"

while :; do
  ts=$(date '+%Y-%m-%d %H:%M:%S')
  proc=$(pgrep -af "local-inference/main.py" || true)
  if [[ -z "$proc" ]]; then
    idle=$((idle+1))
  else
    idle=0
  fi

  {
    echo ""
    echo "=== ${ts} ==="
    if [[ -n "$proc" ]]; then
      echo "process:"
      echo "  $proc"
      ps -o pid,etime,pcpu,pmem,cmd -p "$(pgrep -f 'local-inference/main.py' | head -1)" 2>/dev/null | tail -n +2 | sed 's/^/  /'
    else
      echo "process: (none running)"
    fi
    python3.12 - "$DB" <<'PY'
import sqlite3, sys
db = sqlite3.connect(sys.argv[1])
print("rows per run:")
for r in db.execute(
    "SELECT run_id, COUNT(*) n, SUM(solved) solved, SUM(escalated) esc, "
    "ROUND(SUM(total_cost),6) cost, MAX(created_at) last_row "
    "FROM problem_solving GROUP BY run_id ORDER BY run_id"
):
    print(f"  run_id={r[0]:<18} n={r[1]:<4} solved={r[2]:<4} esc={r[3]:<3} cost={r[4]} last={r[5]}")
PY
  } >> "$LOG" 2>&1

  if (( idle >= IDLE_LIMIT )); then
    echo "" >> "$LOG"
    echo "monitor: no local-inference process for ${IDLE_LIMIT} checks, exiting." >> "$LOG"
    exit 0
  fi
  sleep "$INTERVAL"
done
