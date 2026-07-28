#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PY=./.venv/bin/python

# З аргументами — прямий прохід у CLI: ./run.sh --query "..." або ./run.sh --demo hitl
if [ "$#" -gt 0 ]; then
  exec "$PY" -u -m plan_agent.main "$@"
fi

"$PY" -u -m plan_agent.main --demo plan
"$PY" -u -m plan_agent.main --demo rag
"$PY" -u -m plan_agent.main --demo hitl

# Persistence — два ОКРЕМІ процеси. Перший «падає» посередині плану, другий читає
# стан з agent_state.db і доводить план до кінця. В одному процесі це було б несправжнім.
"$PY" -u -m plan_agent.main --demo persistence-start
"$PY" -u -m plan_agent.main --demo persistence-resume

"$PY" -u -m plan_agent.main --demo threads
