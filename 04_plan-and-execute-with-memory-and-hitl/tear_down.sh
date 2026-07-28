#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Видаляє все згенероване. agent_state.db і demo_results.json — артефакти здачі,
# тож комітити їх треба до запуску цього скрипта.
rm -rf .venv .pytest_cache chroma_db agent_state.db demo_results.json
find . -name __pycache__ -type d -prune -exec rm -rf {} +

echo "Прибрано."
