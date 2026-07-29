"""Моделі, шляхи, ліміти та політики доступу — одне місце для змін."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent

load_dotenv(ROOT / ".env")

# OpenAI-сумісний API OpenRouter.
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "deepseek/deepseek-v4-flash"
EMBED_MODEL = "qwen/qwen3-embedding-8b"
API_KEY_ENV = "OPEN_ROUTER_API_KEY"
# CrewAI ходить через LiteLLM — там модель називається інакше.
LITELLM_MODEL = f"openrouter/{MODEL}"

MCP_SERVER = ROOT / "mcp_server.py"
DB_PATH = ROOT / "agent_state.db"
CHROMA_PATH = str(ROOT / "chroma_db")
COLLECTION_NAME = "nexora_support_kb"
TRAJECTORY_FILE = ROOT / "trajectory.json"
RESULTS_FILE = ROOT / "demo_results.json"
TRACE_FILE = ROOT / "trace.json"
EVAL_FILE = ROOT / "eval_results.json"
RED_TEAM_FILE = ROOT / "red_team_results.json"

AGENTS = ("billing", "tech", "researcher", "general")

# Tool guardrail: кожен агент бачить лише свої інструменти.
# Супервізор не має жодного — маршрутизація не потребує даних.
ALLOWLIST = {
    "supervisor": set(),
    "billing": {"get_ticket", "search_tickets", "get_customer", "get_billing_summary",
                "refund_estimate", "update_ticket_status"},
    "tech": {"get_ticket", "search_tickets", "diagnose_error_code"},
    "researcher": {"search_knowledge"},
    "general": set(),
}

# Незворотні дії — лише після підтвердження людини.
RISKY_TOOLS = {"update_ticket_status"}

MAX_STEPS = 5           # кроків ReAct-циклу всередині одного агента
TIMEOUT_SECONDS = 90.0  # загальний час одного ReAct-циклу
MAX_ITERATIONS = 4      # ітерацій executor-а у Plan-and-Execute
MAX_HANDOFFS = 3        # передач між агентами за один запит
RATE_LIMIT = (30, 60)   # запитів за секунд, per session_id

LLM_TIMEOUT = 90.0
LLM_RETRIES = 2

# $ за 1M токенів (input, output) зі сторінки моделі на openrouter.ai.
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
    """LiteLLM (CrewAI) читає власну змінну — підставляємо той самий ключ."""
    os.environ.setdefault("OPENROUTER_API_KEY", os.environ[API_KEY_ENV])


def make_saver(path=DB_PATH):
    """AsyncSqliteSaver: HITL-пауза і стан MAS переживають перезапуск процесу."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    return AsyncSqliteSaver.from_conn_string(str(path))
