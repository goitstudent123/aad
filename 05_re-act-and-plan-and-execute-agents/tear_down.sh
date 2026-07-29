#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Артефакти здачі (trajectory.json, demo_results.json, graphs.md) теж зникають,
# тож зберігай їх до запуску цього скрипта.
rm -rf .venv .pytest_cache chroma_db agent_state.db agent_state_async.db \
       demo_results.json trajectory.json graphs.md
find . -name __pycache__ -type d -prune -exec rm -rf {} +

echo "Прибрано."
