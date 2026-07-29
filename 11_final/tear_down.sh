#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if command -v docker >/dev/null; then
  docker compose down -v --remove-orphans
fi

# Артефакти здачі теж зникають — збережи їх до запуску.
rm -rf .venv .pytest_cache chroma_db agent_state.db \
       trajectory.json demo_results.json eval_results.json red_team_results.json \
       trace.json langfuse_trace.json test_results.txt
find . -name __pycache__ -type d -prune -exec rm -rf {} +

echo "Прибрано."
