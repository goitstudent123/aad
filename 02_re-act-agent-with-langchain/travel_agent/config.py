"""Налаштування провайдера LLM та мережі."""

import os

from dotenv import load_dotenv

load_dotenv()

# Провайдер: OpenAI-сумісний API OpenRouter (клас ChatOpenAI з LangChain).
BASE_URL = "https://openrouter.ai/api/v1"
MODEL = "xiaomi/mimo-v2.5"
API_KEY_ENV = "OPEN_ROUTER_API_KEY"

# Захисні механізми за замовчуванням.
MAX_STEPS = 6
TIMEOUT_SECONDS = 90.0

# Тайм-аут окремого HTTP-запиту до зовнішнього API.
HTTP_TIMEOUT = 10.0


def make_llm(temperature: float = 0.0):
    """Створює клієнт LLM. Імпорт усередині — щоб тести з фейковою моделлю не тягнули мережу."""
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=MODEL,
        base_url=BASE_URL,
        api_key=os.environ[API_KEY_ENV],
        temperature=temperature,
    )
