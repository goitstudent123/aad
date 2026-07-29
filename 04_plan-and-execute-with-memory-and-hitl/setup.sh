#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-$(command -v python3.13 || command -v python3)}"

"$PYTHON" -m venv .venv
./.venv/bin/pip install --quiet --upgrade pip
./.venv/bin/pip install --quiet -r requirements.txt

if [ ! -f .env ]; then
  printf 'OPEN_ROUTER_API_KEY=\n' > .env
  echo "Створено .env — впиши туди OPEN_ROUTER_API_KEY."
fi

echo "Готово. Далі: ./run.sh"
