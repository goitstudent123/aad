#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-$(command -v python3.13 || command -v python3)}"

"$PYTHON" -m venv .venv
./.venv/bin/pip install --quiet --upgrade pip
./.venv/bin/pip install --quiet -r requirements.txt

if [ ! -f .env ]; then
  cat > .env <<'ENV'
OPEN_ROUTER_API_KEY=

# Langfuse: ці ж значення читає docker-compose.yml і створює з ними проєкт при першому старті.
LANGFUSE_HOST=http://localhost:3000
LANGFUSE_PUBLIC_KEY=pk-lf-nexora-local
LANGFUSE_SECRET_KEY=sk-lf-nexora-local
LANGFUSE_INIT_ORG_ID=nexora
LANGFUSE_INIT_ORG_NAME=Nexora
LANGFUSE_INIT_PROJECT_ID=nexora-helpdesk
LANGFUSE_INIT_PROJECT_NAME=nexora-helpdesk
LANGFUSE_INIT_PROJECT_PUBLIC_KEY=pk-lf-nexora-local
LANGFUSE_INIT_PROJECT_SECRET_KEY=sk-lf-nexora-local
LANGFUSE_INIT_USER_EMAIL=admin@nexora.ua
LANGFUSE_INIT_USER_NAME=Admin
LANGFUSE_INIT_USER_PASSWORD=nexora-local-pass
ENV
  echo "Створено .env — впиши туди OPEN_ROUTER_API_KEY."
fi

# Langfuse піднімається окремо: 6 контейнерів і кілька гігабайтів образів потрібні не всім.
if [ "${1:-}" = "--with-langfuse" ]; then
  docker compose up -d
  echo "Langfuse: http://localhost:3000 (admin@nexora.ua / nexora-local-pass)"
fi

echo "Готово. Далі: ./run.sh"
