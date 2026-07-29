"""База знань агронома: ChromaDB + інструмент search_knowledge (agentic RAG).

У базі — довідкові агротехнічні норми, які майже не змінюються. Живих даних поля
(температура ґрунту, опади, курс) тут свідомо немає: за ними агент має йти в
soil_forecast та input_cost. Рішення «база знань чи інструмент» ухвалює сама модель
за описами інструментів, а не наш код.

Embeddings беремо з OpenRouter, а не з вбудованої в Chroma моделі: так не тягнеться
~80 МБ onnx при першому запуску і всі звернення до моделей ідуть одним шляхом.
"""

import json
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
    "Сівба кукурудзи починається, коли температура ґрунту на глибині 5-10 см стабільно "
    "тримається на рівні 10-12 °C і прогноз не обіцяє повернення холоду. За 8 °C насіння "
    "лежить у ґрунті тижнями й частина його гине від пліснявіння. Глибина заробки 5-7 см.",
    "Озиму пшеницю в центральних областях України сіють з 15 вересня по 5 жовтня: до входу "
    "в зиму рослина має утворити 3-4 пагони, а на це потрібно 45-55 днів з сумою "
    "ефективних температур близько 480 °C. Пізніша сівба знижує зимостійкість.",
    "Обприскування гербіцидом дозволене за швидкості вітру до 5 м/с (18 км/год), "
    "температури повітря 10-25 °C і відсутності опадів у наступні 4-6 годин. За вітру понад "
    "5 м/с робочий розчин зносить на суміжні поля, за спеки понад 25 °C він випаровується "
    "з листка, не встигаючи подіяти.",
    "Дощ протягом двох годин після обробки змиває більшість препаратів і обробку доводиться "
    "повторювати — це подвійні витрати й додаткове пестицидне навантаження. Дощ через 6 годин "
    "після внесення на дію системних препаратів уже майже не впливає.",
    "Період очікування (строк, після якого поле безпечне для людей і збирання) залежить від "
    "препарату: гліфосати — 30 днів до збирання, піретроїди — 20 днів, більшість фунгіцидів — "
    "14-30 днів. Робота людей на обробленому полі дозволена не раніше ніж через 3-7 днів.",
    "Норма азоту під озиму пшеницю на врожай 5-6 т/га становить 120-150 кг д.р./га і ділиться "
    "на 2-3 внесення: рання весняна підживка по мерзлоталому ґрунту, вихід у трубку, колосіння. "
    "Одноразове внесення всієї норми дає надлишковий ріст листя і вилягання.",
    "Вологість ґрунту в шарі 3-9 см нижче 0.15 м³/м³ означає посушливий стан: сівба можлива, "
    "але сходи будуть нерівними. Оптимум для сівби — 0.20-0.30 м³/м³. Понад 0.35 м³/м³ "
    "ґрунт перезволожений, техніка утрамбовує його і руйнує структуру.",
    "Полив за краплинним зрошенням розраховують за випаровуванням (ET0): за ET0 4-5 мм/добу "
    "польові культури потребують близько 25-30 мм на тиждень з урахуванням опадів. Один "
    "міліметр — це 10 м³ води на гектар.",
    "Економічний порогів шкодочинності (ЕПШ) — межа, після якої обробка себе виправдовує: "
    "злакові попелиці — 5-10 особин на стебло у фазі колосіння, клоп шкідлива черепашка — "
    "2 особини на м², шведська муха — 30-50 на 100 помахів сачком. Нижче ЕПШ обробка "
    "лише спалює гроші й вбиває корисних комах.",
    "Заморозки до -1 °C кукурудза у фазі 3-5 листків переживає з пошкодженням листя, точка "
    "росту під землею і рослина відростає. Соняшник у тій же фазі витримує до -4 °C. "
    "Пшениця у фазі цвітіння гине вже за -1 °C — саме тому травневі заморозки небезпечніші "
    "за квітневі.",
    "Зберігання засобів захисту рослин: окреме замкнене приміщення, температура від 0 до "
    "+30 °C, оригінальна тара з етикеткою. Розведений робочий розчин зберігати не можна — "
    "його готують на один виїзд і використовують протягом дня.",
]

DOC_IDS = [f"doc_{i}" for i in range(len(DOCUMENTS))]


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
    trace("knowledge", f"база знань готова: {store.count()} документів у {CHROMA_PATH}")
    return store


class SearchArgs(BaseModel):
    query: str = Field(
        description="Пошуковий запит природною мовою, наприклад 'умови для обприскування'"
    )
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
    """Шукає агротехнічні норми у базі знань господарства.

    Використовуй для правил і нормативів: строки й умови сівби, температура та вологість
    ґрунту для робіт, умови обприскування, періоди очікування, норми добрив, ЕПШ шкідників,
    стійкість культур до заморозків, зберігання препаратів, розрахунок поливу.

    НЕ використовуй для живих даних поля (soil_forecast), курсу валют (input_cost)
    і для дій (schedule_spraying).

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
        return json.dumps(
            {"status": "error", "error": f"база знань недоступна: {exc}"}, ensure_ascii=False
        )
    docs = (found.get("documents") or [[]])[0]
    if not docs:
        return json.dumps(
            {"status": "error", "error": "у базі знань немає нічого релевантного"},
            ensure_ascii=False,
        )
    trace("knowledge", f"знайдено {len(docs)} документів, перший: {short(docs[0], 120)}")
    return json.dumps(
        {"status": "ok", "data": {"documents": docs, "source": "knowledge_base"}},
        ensure_ascii=False,
    )
