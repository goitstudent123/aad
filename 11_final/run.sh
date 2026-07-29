#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PY=./.venv/bin/python

# ./run.sh --demo hitl | --query "..." | --demo all (за замовчуванням)
if [ "$#" -gt 0 ]; then
  exec "$PY" -u main.py "$@"
fi

"$PY" -u main.py --demo all
