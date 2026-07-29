"""Провайдер LLM, сховища стану й знань, ліміти агента."""

import os
import sqlite3

from dotenv import load_dotenv

load_dotenv()

# OpenAI-сумісний API OpenRouter.
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "deepseek/deepseek-v4-flash"
# Друга модель — тільки для бонусного порівняння провайдерів (--demo providers).
ALT_MODEL = "google/gemini-2.5-flash-lite"
PROVIDERS = [MODEL, ALT_MODEL]
EMBED_MODEL = "qwen/qwen3-embedding-8b"
API_KEY_ENV = "OPEN_ROUTER_API_KEY"

# Файл, не :memory: — інакше стан не переживе перезапуск процесу.
DB_PATH = "agent_state.db"

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "agro_knowledge"

# Захисні механізми ReAct-циклу.
MAX_STEPS = 10
TIMEOUT_SECONDS = 120.0

# Скільки разів executor може відпрацювати в межах одного плану: без цього
# replan → executor → replan крутиться вічно.
MAX_ITERATIONS = 8

# Ліміт вкладеного ReAct-агента, який виконує один крок плану.
STEP_MAX_STEPS = 4
STEP_TIMEOUT_SECONDS = 60.0

HTTP_TIMEOUT = 10.0

# Без тайм-ауту httpx чекає відповіді провайдера вічно, і агент «зависає» посеред кроку.
LLM_TIMEOUT = 90.0
LLM_RETRIES = 2


def make_llm(model: str = MODEL, temperature: float = 0.0):
    """Клієнт LLM. Імпорт усередині — щоб тести з фейковою моделлю не тягнули мережу."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        base_url=BASE_URL,
        api_key=os.environ[API_KEY_ENV],
        temperature=temperature,
        timeout=LLM_TIMEOUT,
        max_retries=LLM_RETRIES,
    )


def make_saver(path: str = DB_PATH):
    """SqliteSaver на файлі. check_same_thread=False — LangGraph ходить із кількох потоків."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    return SqliteSaver(sqlite3.connect(path, check_same_thread=False))
