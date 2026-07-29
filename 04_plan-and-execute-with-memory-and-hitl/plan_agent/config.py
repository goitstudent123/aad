"""Налаштування провайдера LLM, сховищ стану/знань та лімітів.

Одне місце для змін: модель, embedding-модель, шляхи до agent_state.db і chroma_db.
"""

import os
import sqlite3

from dotenv import load_dotenv

load_dotenv()

# Провайдер: OpenAI-сумісний API OpenRouter (клас ChatOpenAI з LangChain).
BASE_URL = "https://openrouter.ai/api/v1"
# xiaomi/mimo-v2.5 постійно віддавав 429 «temporarily rate-limited upstream», через що
# прогін демо тягнувся десятки хвилин на самих лише повторах.
MODEL = "deepseek/deepseek-v4-flash"
EMBED_MODEL = "qwen/qwen3-embedding-8b"
API_KEY_ENV = "OPEN_ROUTER_API_KEY"

# Persistence: ФАЙЛ, не :memory: — інакше стан не переживе перезапуск процесу.
DB_PATH = "agent_state.db"

# Vector store для RAG.
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "travel_knowledge"

# Запобіжник: скільки разів executor може відпрацювати в межах одного прогону.
# Без нього replan → executor → replan може крутитися вічно.
MAX_ITERATIONS = 8

# Тайм-аут окремого HTTP-запиту до зовнішнього API.
HTTP_TIMEOUT = 10.0

# Тайм-аут одного запиту до LLM і кількість власних повторів клієнта.
# За замовчуванням у ChatOpenAI тайм-аут відсутній: коли OpenRouter приймає з'єднання й
# не відповідає, httpx чекає у socket.recv вічно і агент «зависає» посеред кроку без жодного
# рядка в лозі. З тайм-аутом це стає звичайною помилкою, яку демо ловлять і повторюють.
LLM_TIMEOUT = 90.0
LLM_RETRIES = 2


def make_llm(temperature: float = 0.0):
    """Створює клієнт LLM. Імпорт усередині — щоб тести з фейковою моделлю не тягнули мережу."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=MODEL,
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
