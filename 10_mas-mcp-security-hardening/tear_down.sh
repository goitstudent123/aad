#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if command -v docker >/dev/null; then
  docker compose down -v --remove-orphans
fi

# Артефакти здачі (demo_results.json, trace.json, langfuse_trace.json) теж зникають —
# збережи їх до запуску.
rm -rf .venv .pytest_cache .deepeval* agent_state.db \
       demo_results.json trace.json langfuse_trace.json test_results.txt
find . -name __pycache__ -type d -prune -exec rm -rf {} +

echo "Прибрано."
