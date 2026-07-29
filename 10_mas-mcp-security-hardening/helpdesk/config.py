"""Моделі, шляхи, ліміти та політики доступу — одне місце для змін."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent

load_dotenv(ROOT / ".env")

# OpenAI-сумісний API OpenRouter.
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "deepseek/deepseek-v4-flash"
API_KEY_ENV = "OPEN_ROUTER_API_KEY"
# CrewAI, ADK та deepeval ходять через LiteLLM — там модель називається інакше.
LITELLM_MODEL = f"openrouter/{MODEL}"

MCP_SERVER = ROOT / "mcp_server.py"
DB_PATH = ROOT / "agent_state.db"
TRACE_FILE = ROOT / "trace.json"
RESULTS_FILE = ROOT / "demo_results.json"

AGENTS = ("kb_agent", "directory_agent", "ops_agent")

# Tool guardrail: кожен агент бачить лише свої інструменти.
ALLOWLIST = {
    "kb_agent": {"search_kb"},
    "directory_agent": {"lookup_employee", "list_tickets"},
    "ops_agent": {"lookup_employee", "reset_password"},
}

# Незворотні дії — лише після підтвердження людини.
RISKY_TOOLS = {"reset_password"}

MAX_HANDOFFS = 6
LLM_TIMEOUT = 90.0
LLM_RETRIES = 2

# $ за 1M токенів (input, output) зі сторінки моделі на openrouter.ai — оновити при зміні моделі.
PRICE_PER_MTOK = (0.28, 0.42)


def make_llm(model: str = MODEL, temperature: float = 0.0):
    """Клієнт LLM. Імпорт усередині — щоб офлайн-тести не тягнули langchain_openai."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        base_url=BASE_URL,
        api_key=os.environ[API_KEY_ENV],
        temperature=temperature,
        timeout=LLM_TIMEOUT,
        max_retries=LLM_RETRIES,
    )


def litellm_env() -> None:
    """LiteLLM (CrewAI, ADK, deepeval) читає власну змінну — підставляємо той самий ключ."""
    os.environ.setdefault("OPENROUTER_API_KEY", os.environ[API_KEY_ENV])


def make_saver(path=DB_PATH):
    """AsyncSqliteSaver на файлі (граф асинхронний): HITL-пауза переживає перезапуск процесу."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    return AsyncSqliteSaver.from_conn_string(str(path))
