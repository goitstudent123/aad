"""База знань агента: ChromaDB + інструмент search_knowledge (Agentic RAG).

У базі — довідкові, майже незмінні факти домену: візи, страховки, правила бронювання,
типові ціни. Живих даних (погода, курс валют) тут свідомо немає — за ними агент має
йти в geocode/weather/currency. Саме на цьому й видно agentic RAG: рішення «база знань
чи інструмент» ухвалює сама модель за docstring-ами, а не наш код.

Embeddings беремо з OpenRouter (qwen/qwen3-embedding-8b), а не з вбудованої в Chroma
all-MiniLM-L6-v2: правила репозиторію кажуть ходити по моделі через OpenRouter, до того ж
так не потрібно тягнути ~80 МБ onnx-моделі при першому запуску.
"""

import os
from functools import lru_cache

from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from .config import (
    API_KEY_ENV,
    BASE_URL,
    CHROMA_PATH,
    COLLECTION_NAME,
    EMBED_MODEL,
    LLM_RETRIES,
    LLM_TIMEOUT,
)
from .logs import short, trace

DOCUMENTS = [
    "Громадянам України для коротких поїздок до Шенгенської зони діє безвіз: до 90 днів "
    "протягом кожних 180. Потрібен біометричний паспорт, дійсний ще щонайменше три місяці "
    "після виїзду. Прикордонник може попросити зворотний квиток і підтвердження житла.",
    "Медична страховка для Шенгену обов'язкова: покриття щонайменше 30 000 EUR, дійсна на "
    "всі дні поїздки. Базовий поліс на тиждень коштує 8-15 EUR. Без страховки на кордоні "
    "можуть відмовити у в'їзді, а лікування без неї обходиться в сотні євро за візит.",
    "Типові ціни на житло у Центральній Європі за ніч: хостел (ліжко у спільній кімнаті) "
    "15-30 EUR, готель 3* 60-110 EUR, апартаменти на двох 70-130 EUR. У Львові та Кракові "
    "дешевше на 20-30%, ніж у Празі чи Відні. Ціни зростають на 40-60% у високий сезон.",
    "Правила бронювання готелю: безкоштовне скасування зазвичай доступне до 24-48 годин до "
    "заїзду. Тариф non-refundable дешевший на 10-20%, але гроші не повертають зовсім. "
    "Стандартний час заїзду — після 14:00, виїзду — до 11:00. Депозит за інциденти "
    "блокують на картці, зазвичай 50-100 EUR.",
    "Туристичний збір (city tax) у Європі платять окремо від бронювання, на місці: 1-5 EUR "
    "з людини за ніч залежно від міста й категорії готелю. У Кракові це близько 2 EUR, у "
    "Празі — 50 CZK, у Відні — 3.2% від вартості проживання.",
    "Найкращі місяці для міських поїздок Європою — квітень-червень і вересень-жовтень: "
    "тепло, але без спеки й натовпу. Липень-серпень — найдорожчий і найбільш людний період. "
    "Січень-березень найдешевші, проте частина музеїв працює за скороченим розписом.",
    "Ручна поклажа у бюджетних авіакомпаніях: одна невелика сумка під сидіння входить у "
    "тариф, валіза у верхню полицю — за окрему плату 15-40 EUR, дешевше при купівлі "
    "заздалегідь, ніж в аеропорту. Ліміт рідин — ємності до 100 мл у прозорому пакеті до 1 л.",
    "Транспорт у європейських містах: одноразовий квиток на метро/трамвай 1.5-3 EUR, "
    "денний проїзний 6-16 EUR і майже завжди вигідніший за три поїздки. Квиток треба "
    "компостувати перед поїздкою, штраф за безбілетний проїзд — 50-120 EUR.",
    "Гроші в поїздці: у Польщі, Чехії та Угорщині своя валюта, тому платити краще картою "
    "у місцевій валюті — конвертація банку зазвичай вигідніша за курс обмінників в "
    "аеропорту. Обмінники в центрі міста дають курс на 3-7% кращий за аеропортові. "
    "Знімати готівку в банкоматах чужих мереж — комісія 3-5 EUR за операцію.",
    "Безпека туриста: найпоширеніша проблема — кишенькові злодії в метро, на вокзалах і "
    "біля головних пам'яток. Документи й основну частину грошей краще тримати в готелі, з "
    "собою — копію паспорта. Єдиний номер екстрених служб у ЄС — 112, працює безкоштовно "
    "з будь-якої SIM-картки.",
]

DOC_IDS = [f"doc_{i}" for i in range(len(DOCUMENTS))]


def embed(texts: list[str]) -> list[list[float]]:
    """Векторизує тексти через OpenRouter. Імпорт усередині — тести сюди не доходять."""
    from openai import OpenAI

    # Тайм-аут обов'язковий з тієї ж причини, що й у make_llm: без нього зависання
    # провайдера вішає весь граф.
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
    """Persistent-колекція Chroma; при першому виклику наповнює її документами.

    PersistentClient (а не Client()) — щоб embeddings не перераховувалися щозапуску.
    """
    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    store = client.get_or_create_collection(COLLECTION_NAME)
    if store.count() != len(DOCUMENTS):
        trace("knowledge", f"наповнюю базу знань: {len(DOCUMENTS)} документів через {EMBED_MODEL}")
        store.upsert(documents=DOCUMENTS, ids=DOC_IDS, embeddings=embed(DOCUMENTS))
    trace("knowledge", f"база знань готова: {store.count()} документів у {CHROMA_PATH}")
    return store


class SearchArgs(BaseModel):
    query: str = Field(description="Пошуковий запит природною мовою, наприклад 'страховка Шенген'")
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
    """Шукає довідкову інформацію у базі знань про подорожі.

    Використовуй для правил, вимог і типових цін: візи та безвіз, страховка, правила
    бронювання й скасування, туристичний збір, ручна поклажа, транспорт, сезонність,
    поради щодо грошей і безпеки.

    НЕ використовуй для живих даних (погода, курс валют) і для дій (бронювання) —
    для них є окремі інструменти.

    Args:
        query: Пошуковий запит.
        n_results: Скільки документів повернути (за замовчуванням 3).

    Returns:
        Найрелевантніші документи з бази знань, розділені рядком '---'.
    """
    trace("knowledge", f"RAG-запит: {short(query, 120)} (топ-{n_results})")
    try:
        found = collection().query(query_embeddings=embed([query]), n_results=n_results)
    except Exception as exc:  # noqa: BLE001 — Chroma чи мережа; агент має дізнатися причину
        trace("knowledge", f"⚠ база знань недоступна: {short(exc)}")
        return f"error: база знань недоступна ({exc})"
    docs = (found.get("documents") or [[]])[0]
    if not docs:
        trace("knowledge", "нічого релевантного не знайдено")
        return "error: у базі знань немає нічого релевантного до цього запиту"
    trace("knowledge", f"знайдено {len(docs)} документів, перший: {short(docs[0], 120)}")
    return "\n---\n".join(docs)
