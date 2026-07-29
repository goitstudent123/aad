#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PY=./.venv/bin/python

# З аргументами — прямий прохід у CLI: ./run.sh --query "..." або ./run.sh --demo hitl
if [ "$#" -gt 0 ]; then
  exec "$PY" -u -m agro_agent.main "$@"
fi

"$PY" -u -m agro_agent.main --demo graph
"$PY" -u -m agro_agent.main --demo fallback
"$PY" -u -m agro_agent.main --demo react
"$PY" -u -m agro_agent.main --demo plan
"$PY" -u -m agro_agent.main --demo rag
"$PY" -u -m agro_agent.main --demo hitl
"$PY" -u -m agro_agent.main --demo memory

# Persistence — два ОКРЕМІ процеси. Перший зупиняється перед ризиковою дією, другий
# читає стан з agent_state.db і доводить план до кінця. В одному процесі це було б
# несправжнім відновленням.
"$PY" -u -m agro_agent.main --demo persistence-start
"$PY" -u -m agro_agent.main --demo persistence-resume

"$PY" -u -m agro_agent.main --demo async
"$PY" -u -m agro_agent.main --demo compare
"$PY" -u -m agro_agent.main --demo providers
