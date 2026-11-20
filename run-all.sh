#!/usr/bin/env bash
# Build, test and validate AI Desk end to end on the KodeBox, then print a PASS/FAIL table.
#   ./run-all.sh              everything (spends roughly 130k KodeKey tokens on a cold cache, far less on re-runs)
#   ./run-all.sh --offline    only the free steps (setup, offline tests, lint and security scans)
#   ./run-all.sh --probe      also re-run the KodeKey model probe (about 2k tokens)
set -uo pipefail
cd "$(dirname "$0")"

OFFLINE=0; PROBE=0
for arg in "$@"; do
  case "$arg" in
    --offline) OFFLINE=1 ;;
    --probe) PROBE=1 ;;
    *) echo "unknown option: $arg"; exit 2 ;;
  esac
done
export AIDESK_DAILY_TOKEN_CAP="${AIDESK_DAILY_TOKEN_CAP:-250000}"
mkdir -p var reports/live
LOG="reports/live/run-all.log"; : > "$LOG"
declare -a NAMES RESULTS TIMES
START_ALL=$(date +%s)

step() {
  local name="$1"; shift
  local t0; t0=$(date +%s)
  echo "=== $name" | tee -a "$LOG"
  if "$@" >>"$LOG" 2>&1; then RESULTS+=("PASS"); else RESULTS+=("FAIL"); fi
  NAMES+=("$name"); TIMES+=("$(( $(date +%s) - t0 ))s")
  echo "    ${RESULTS[-1]} (${TIMES[-1]})"
}

step "setup: locked environment"            ./setup/bootstrap.sh
step "offline tests (fake model)"           uv run pytest
step "lint + security rules (ruff)"         uv run ruff check src tests setup
step "static security scan (bandit)"        uv run bandit -q -r src -ll
step "dependency audit (pip-audit)"         bash -c 'uv export --frozen --no-dev --no-emit-project --format requirements-txt > var/req.txt && uv run pip-audit -r var/req.txt --disable-pip --no-deps'

if [ "$OFFLINE" -eq 0 ]; then
  [ "$PROBE" -eq 1 ] && step "KodeKey probe"  uv run python setup/probe_kodekey.py
  step "index knowledge base"               uv run aidesk ingest
  step "live agents (KodeKey + web)"        uv run pytest -m live tests/live -s
  step "support evals (quality gate)"       uv run pytest -m live tests/evals -s
  step "start services (tmux)"              ./setup/start-services.sh
  step "publish /app/ via nginx"            ./setup/publish-app.sh
  step "end to end through nginx, MCP, A2A" uv run pytest -m e2e tests/e2e -s
fi

echo
printf '%-40s %-6s %s\n' "STEP" "RESULT" "TIME"
for i in "${!NAMES[@]}"; do printf '%-40s %-6s %s\n' "${NAMES[$i]}" "${RESULTS[$i]}" "${TIMES[$i]}"; done
echo "Total: $(( $(date +%s) - START_ALL ))s. Full log: $LOG"
uv run aidesk usage 2>/dev/null | grep -E '"tokens"|"day"' | head -2
[[ " ${RESULTS[*]} " == *" FAIL "* ]] && exit 1 || exit 0
