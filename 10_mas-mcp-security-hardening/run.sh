#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PY=./.venv/bin/python

# З аргументами — прямий прохід у CLI: ./run.sh --demo hitl або ./run.sh --query "..."
if [ "$#" -gt 0 ]; then
  exec "$PY" -u -m helpdesk.main "$@"
fi

"$PY" -u -m helpdesk.main --demo all
