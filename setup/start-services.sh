#!/usr/bin/env bash
# Start the AI Desk services in a tmux session called "aidesk", all bound to 127.0.0.1:
#   api 8000 (web app + UI), mcp 8001 (/mcp, Streamable HTTP), a2a 8002 (Deep Research agent)
# Runs as your user: no systemd units are created. Run again after the server restarts.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
BIN="$REPO/.venv/bin/aidesk"
[ -x "$BIN" ] || { echo "Run ./setup/bootstrap.sh first"; exit 1; }

# tmux sessions don't inherit this shell's variables, so pass the token caps explicitly.
CAPS="AIDESK_DAILY_TOKEN_CAP=${AIDESK_DAILY_TOKEN_CAP:-150000} AIDESK_RUN_TOKEN_CAP=${AIDESK_RUN_TOKEN_CAP:-40000}"
tmux kill-session -t aidesk 2>/dev/null || true
"$BIN" ingest >/dev/null
tmux new-session -d -s aidesk -n api "env $CAPS AIDESK_A2A_URL=http://127.0.0.1:8002 $BIN serve --port 8000 2>&1 | tee -a $REPO/var/api.log"
tmux new-window -t aidesk -n mcp "env $CAPS $BIN mcp --http --port 8001 2>&1 | tee -a $REPO/var/mcp.log"
tmux new-window -t aidesk -n a2a "env $CAPS $BIN a2a --port 8002 2>&1 | tee -a $REPO/var/a2a.log"

for port in 8000 8001 8002; do
  for _ in $(seq 1 40); do
    if (exec 3<>/dev/tcp/127.0.0.1/$port) 2>/dev/null; then echo "port $port: up"; break; fi
    sleep 0.5
  done
done
curl -fsS http://127.0.0.1:8000/api/health >/dev/null && echo "api health: ok"
curl -fsS http://127.0.0.1:8002/.well-known/agent-card.json >/dev/null && echo "a2a agent card: ok"
echo "Attach with: tmux attach -t aidesk   (detach: Ctrl-b d)"
