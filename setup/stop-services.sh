#!/usr/bin/env bash
# Stop the AI Desk services started by start-services.sh.
tmux kill-session -t aidesk 2>/dev/null && echo "stopped" || echo "not running"
