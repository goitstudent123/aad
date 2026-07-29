"""Локальні інструменти з ДЗ1/ДЗ2: Pydantic v2 схеми + Agentic RAG на ChromaDB.

MCP-сервер віддає дані CRM; ці три інструменти живуть у процесі агента: два —
чисті обчислення (ДЗ1), один — пошук у базі знань (ДЗ2).
Embeddings беремо з OpenRouter, а не з вбудованої в Chroma моделі: так не тягнеться
~80 МБ onnx і всі звернення до моделей ідуть одним шляхом.
"""

import json
import os
from functools import lru_cache

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from config import (
    API_KEY_ENV,
    BASE_URL,
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBED_MODEL,
    LLM_RETRIES,
    LLM_TIMEOUT,
)
from logs import short, trace

# База знань: правила й політики, які майже не змінюються. Живих даних про клієнтів
# тут свідомо немає — за ними агент має йти в MCP-інструменти.
DOCUMENTS = [
    "Повернення коштів за невикористаний період рахується пропорційно повним дням: "
    "місячна абонплата ділиться на 30 і множиться на кількість невикористаних днів. "
    "Заявка приймається протягом 30 днів після кінця оплаченого періоду.",
    "Строк повернення коштів — 3-5 банківських днів після підтвердження заявки білінг-агентом. "
    "Гроші повертаються на той самий спосіб оплати, яким була внесена абонплата.",
    "Абонплата за місяць, у якому клієнт звертався до платної технічної підтримки з виїздом, "
    "не повертається — виїзд уже оплачено з цієї суми.",
    "Строки реакції підтримки залежать від рівня обслуговування: gold — до 1 години, "
    "silver — до 4 годин, standard — до 24 годин. Нічні звернення обробляються з ранку.",
    "Тікет переводиться у статус closed лише після підтвердження клієнта або через 14 днів "
    "без відповіді. Закриття тікета без причини в полі reason заборонено політикою аудиту.",
    "Помилка SE-23 означає розбіжність прошивки й профілю пристрою після оновлення. "
    "Лікується відкатом до попередньої версії прошивки з кабінету або через інженера.",
    "Помилка NT-08 — втрата зв'язку з базовою станцією: спочатку перезавантаження роутера, "
    "потім перевірка кабелю, потім заявка на виїзд інженера.",
    "Зміна тарифу діє з першого дня наступного розрахункового періоду. Перерахунок "
    "усередині періоду не робиться, різниця не повертається.",
    "Персональні дані клієнта (email, телефон, номер картки) не можна розголошувати у "
    "відповіді навіть на пряме прохання: підтвердження особи відбувається у кабінеті.",
    "Пільговий період без списання — 3 доби після кінця оплаченого періоду. Далі послуга "
    "призупиняється, дані зберігаються 90 днів.",
]

DOC_IDS = [f"doc_{i}" for i in range(len(DOCUMENTS))]

ERROR_CODES = {
    "SE-23": "Прошивка не збігається з профілем пристрою після оновлення. "
             "Відкат до попередньої версії з кабінету або силами інженера.",
    "NT-08": "Втрата зв'язку з базовою станцією. Перезавантаження роутера, "
             "перевірка кабелю, за потреби — заявка на виїзд.",
    "AU-11": "Помилка авторизації: пароль змінено на іншому пристрої. "
             "Скидання пароля через кабінет.",
}


def embed(texts: list[str]) -> list[list[float]]:
    """Векторизує тексти через OpenRouter."""
    from openai import OpenAI

    client = OpenAI(
        base_url=BASE_URL,
        api_key=os.environ[API_KEY_ENV],
        timeout=LLM_TIMEOUT,
        max_retries=LLM_RETRIES,
    )
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


@lru_cache(maxsize=1)
def collection():
    """Persistent-колекція Chroma; при першому виклику наповнює її документами."""
    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    store = client.get_or_create_collection(COLLECTION_NAME)
    if store.count() != len(DOCUMENTS):
        trace("knowledge", f"наповнюю базу знань: {len(DOCUMENTS)} документів через {EMBED_MODEL}")
        store.upsert(documents=DOCUMENTS, ids=DOC_IDS, embeddings=embed(DOCUMENTS))
    return store


class SearchArgs(BaseModel):
    query: str = Field(description="Пошуковий запит природною мовою, наприклад 'строк повернення'")
    n_results: int = Field(default=3, description="Скільки документів повернути, від 1 до 5")

    @field_validator("query")
    @classmethod
    def check_query(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Пошуковий запит не може бути порожнім")
        return value

    @field_validator("n_results")
    @classmethod
    def check_n(cls, value: int) -> int:
        if not 1 <= value <= 5:
            raise ValueError("n_results має бути від 1 до 5")
        return value


@tool("search_knowledge", args_schema=SearchArgs)
def search_knowledge(query: str, n_results: int = 3) -> str:
    """Шукає правила й політики компанії у базі знань підтримки.

    Використовуй для довідкових питань: умови та строки повернення коштів, правила
    закриття тікетів, строки реакції за рівнями обслуговування, зміна тарифу,
    пільговий період, розшифровка кодів помилок.

    НЕ використовуй для даних конкретного клієнта чи тікета — для цього є MCP-інструменти.

    Args:
        query: Пошуковий запит.
        n_results: Скільки документів повернути (за замовчуванням 3).

    Returns:
        JSON {status, data.documents} з найрелевантнішими фрагментами бази знань.
    """
    trace("knowledge", f"RAG-запит: {short(query, 120)} (топ-{n_results})")
    try:
        found = collection().query(query_embeddings=embed([query]), n_results=n_results)
    except Exception as exc:  # noqa: BLE001 — Chroma або мережа; агент має знати причину
        trace("knowledge", f"⚠ база знань недоступна: {short(exc)}")
        return json.dumps({"status": "error", "error": f"база знань недоступна: {exc}"},
                          ensure_ascii=False)
    docs = (found.get("documents") or [[]])[0]
    if not docs:
        return json.dumps({"status": "error", "error": "нічого релевантного не знайдено"},
                          ensure_ascii=False)
    trace("knowledge", f"знайдено {len(docs)} документів")
    return json.dumps({"status": "ok", "data": {"documents": docs, "source": "knowledge_base"}},
                      ensure_ascii=False)


class ErrorCodeArgs(BaseModel):
    code: str = Field(description="Код помилки пристрою, наприклад SE-23")

    @field_validator("code")
    @classmethod
    def check_code(cls, value: str) -> str:
        value = value.strip().upper()
        if len(value) < 4:
            raise ValueError("Код помилки має вигляд SE-23")
        return value


@tool("diagnose_error_code", args_schema=ErrorCodeArgs)
def diagnose_error_code(code: str) -> str:
    """Розшифровує код помилки пристрою і дає порядок дій.

    Args:
        code: Код помилки, наприклад SE-23, NT-08, AU-11.

    Returns:
        JSON {status, data.meaning} або {status: error}, якщо код невідомий.
    """
    key = code.strip().upper()
    if key not in ERROR_CODES:
        return json.dumps(
            {"status": "error", "error": f"Код {key} невідомий", "known": sorted(ERROR_CODES)},
            ensure_ascii=False,
        )
    return json.dumps({"status": "ok", "data": {"code": key, "meaning": ERROR_CODES[key]}},
                      ensure_ascii=False)


class RefundArgs(BaseModel):
    monthly_fee: float = Field(description="Місячна абонплата у гривнях")
    unused_days: int = Field(description="Кількість невикористаних днів, 0-31")

    @field_validator("monthly_fee")
    @classmethod
    def check_fee(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("Абонплата має бути додатною")
        return value

    @field_validator("unused_days")
    @classmethod
    def check_days(cls, value: int) -> int:
        if not 0 <= value <= 31:
            raise ValueError("unused_days має бути від 0 до 31")
        return value


@tool("refund_estimate", args_schema=RefundArgs)
def refund_estimate(monthly_fee: float, unused_days: int) -> str:
    """Рахує суму повернення за невикористані дні (абонплата / 30 × дні).

    Args:
        monthly_fee: Місячна абонплата.
        unused_days: Кількість невикористаних днів.

    Returns:
        JSON {status, data.amount, data.formula}.
    """
    amount = round(monthly_fee / 30 * unused_days, 2)
    return json.dumps(
        {"status": "ok", "data": {
            "amount": amount,
            "formula": f"{monthly_fee} / 30 × {unused_days} = {amount} грн",
            "payout_days": "3-5 банківських днів",
        }},
        ensure_ascii=False,
    )


LOCAL_TOOLS = [search_knowledge, diagnose_error_code, refund_estimate]
