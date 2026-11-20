#!/usr/bin/env bash
# Prepare AI Desk on the KodeBox: Python environment from uv.lock, data folder, and a settings check.
# Safe to run again. It never prints your KodeKey key.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v uv >/dev/null || { echo "uv is missing (the AI Engineering image ships it)"; exit 1; }
uv sync --frozen
mkdir -p var reports

uv run python - <<'PY'
from aidesk.config import get_settings
s = get_settings()
print(f"provider : {s.provider}")
print(f"base url : {s.base_url}")
print(f"key set  : {'yes' if s.api_key else 'no'} ({len(s.api_key)} chars)")
print(f"data dir : {s.data_dir}")
for role, model in s.models.items():
    print(f"role {role:<9}: {model}")
PY
