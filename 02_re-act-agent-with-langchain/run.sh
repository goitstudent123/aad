#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

# Без аргументів — прогін усіх тест-кейсів; з --query "..." — один запит.
exec ./.venv/bin/python -u -m travel_agent.main "$@"
