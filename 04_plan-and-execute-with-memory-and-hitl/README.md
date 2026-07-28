# ДЗ №4 — Plan-and-Execute з memory та HITL

Тревел-планер **«Weekend Escape»** із ДЗ №2, переписаний із ReAct на **Plan-and-Execute**:
агент спершу складає повний план, потім виконує його крок за кроком і після кожного кроку
вирішує, чи план досі актуальний. Додано persistence (SqliteSaver), agentic RAG (ChromaDB)
та human-in-the-loop на ризиковій дії.

ReAct агент вирішував наступний крок «на льоту», побачивши
результат інструменту. Тут план існує до першого виклику інструменту, а адаптація до
реальності — окреме рішення окремого вузла (`replanner`).

## Архітектура

```
plan_agent/
  config.py      # провайдер LLM, embedding-модель, шляхи до agent_state.db і chroma_db, ліміти
  schemas.py     # structured outputs: Plan, ReplanDecision, TravelAnswer
  tools.py       # geocode, weather, currency + ризиковий book_hotel; RISKY_TOOLS
  knowledge.py   # ChromaDB (10 документів) + інструмент search_knowledge
  graph.py       # LangGraph: planner → executor → approval → act → replanner
  logs.py        # живий трейс ходу роботи з таймстемпами
  demos.py       # демонстрації всіх чотирьох завдань
  trajectory.py  # запис артефактів
  main.py        # CLI
tests/           # 50 офлайн-тестів (LLM, мережа та Chroma замокані)
```

Граф:

```
START → planner → executor ──(є tool calls)──► approval ──► act ──┐
                     │                                            │
                     └────────(немає tool calls)──────────────────►replanner
                                                                     │
                            ┌────────────(не завершено)─────────────┘
                            ▼
                        executor          (завершено) ──► respond ──► END
```

| Вузол | Що робить |
|---|---|
| `planner` | `with_structured_output(Plan)` → `goal` + `steps[]`. Повний план одним викликом |
| `executor` | бере `plan[current_step]`, просить LLM з `bind_tools` обрати інструменти саме для цього кроку |
| `approval` | HITL-шлюз: на ризиковому інструменті викликає `interrupt()` з деталями дії |
| `act` | виконує підтверджені виклики, відхилені не виконує; пише результат кроку |
| `replanner` | `with_structured_output(ReplanDecision)` → `continue` / `replan` / `finish` |
| `respond` | `with_structured_output(TravelAnswer)`: `summary`, `facts[]`, `actions[]`, `warnings[]` |

**Чому executor розрізаний на три вузли.** `interrupt()` після `resume` перезапускає вузол
**з початку**. Якби виклик LLM і `interrupt()` жили в одному вузлі (як у прикладі з
методички), після кожного «approve» модель викликалася б удруге — зайві токени і, гірше,
шанс отримати інші аргументи, ніж ті, які підтвердила людина. Тому LLM обирає інструменти в
`executor`, підтвердження питає `approval` (він читає лише стан, без LLM), а виконує `act`.
Тест `test_approve_executes_the_tool_without_calling_llm_again` це фіксує.

**Стан** (`PlanExecuteState`): `messages`, `goal`, `plan`, `current_step`, `results`,
`completed`, `iterations`/`max_iterations`, `pending` (tool calls, що чекають на рішення),
`decisions` (рішення людини), `log` (плаский журнал для артефактів), `stop_reason`, `answer`.

**Запобіжник.** `max_iterations=8` на кількість спрацювань `executor`: replanner може
теоретично крутити `replan → executor → replan` вічно. Плюс `recursion_limit` самого
LangGraph як друга лінія.

## Інструменти

| Інструмент | Джерело | Тип |
|---|---|---|
| `geocode` | Open-Meteo Geocoding | назва міста → координати |
| `weather` | Open-Meteo Forecast | живий прогноз на 1–7 днів |
| `currency` | НБУ | сума в валюті → гривні за офіційним курсом |
| `search_knowledge` | ChromaDB | довідка з бази знань (RAG) |
| `book_hotel` | — | **ризикова незворотна дія**, лише через HITL |

Кожен має Pydantic-схему з `Field(description=...)` і щонайменше один `field_validator`
(`nights` 1–30, `check_in` у форматі `YYYY-MM-DD`, `total_cost > 0`, код валюти ISO 4217 тощо).
Помилки мережі повертаються як `{"error": ...}`, помилки валідації в `act` перетворюються на
результат кроку — граф не падає в обох випадках.

## База знань (Agentic RAG)

`plan_agent/knowledge.py`: **10 документів** по 2–4 речення — виключно довідкові, майже
незмінні факти домену: безвіз і документи, медична страховка, типові ціни на житло, правила
скасування бронювання, туристичний збір, сезонність, ручна поклажа, міський транспорт, гроші
та обмін, безпека туриста.

Живих даних у базі **немає свідомо**: погода й курс валют лишаються за `weather`/`currency`.
Саме на цьому видно agentic RAG — рішення «база знань чи інструмент» ухвалює модель за
docstring-ами, а не наш код. Демо `--demo rag` ставить два запити тому самому агенту:
довідковий (очікуємо `search_knowledge`) і про живі дані (очікуємо `geocode`/`weather`/`currency`).

- Vector store: `chromadb.PersistentClient(path="./chroma_db")`, колекція `travel_knowledge`.
- Embeddings: **OpenRouter `qwen/qwen3-embedding-8b`** (правило репозиторію — усе через
  OpenRouter). Вбудована в Chroma `all-MiniLM-L6-v2` не використовується, тож ~80 МБ onnx
  при першому запуску не завантажуються. Векторизуємо самі й передаємо `embeddings=` /
  `query_embeddings=` явно.
- Наповнення ідемпотентне: `upsert` спрацьовує лише коли `count() != len(DOCUMENTS)`.

## Ризиковий інструмент і HITL flow

`book_hotel(hotel_name, city, check_in, nights, total_cost)` — бронює готель і списує гроші,
тобто дію не відкотити. Він у `RISKY_TOOLS`, і `approval` зупиняє на ньому граф:

```
executor обрав book_hotel(...)
  → approval: interrupt({tool, args, step, message})   ← граф стоїть, стан у agent_state.db
  → людина: app.invoke(Command(resume=...), config)
      {'approved': True}                     → act виконує бронювання
      {'approved': False, 'reason': '...'}   → act НЕ виконує, пише «ВІДХИЛЕНО» у результат кроку
      {'approved': True, 'args': {...}}      → act виконує зі ЗМІНЕНИМИ параметрами (edit)
  → replanner працює далі в обох випадках: агент не падає від відмови
```

`resume` приймає також `True/False` і рядок `approve`/`reject` — людина відповідає руками,
тож форма буває різна (`_decision_from` у `graph.py`).

## Persistence

`SqliteSaver(sqlite3.connect("agent_state.db", check_same_thread=False))` — **файл**, не
`:memory:`, інакше стан не переживе перезапуск процесу; `check_same_thread=False` бо LangGraph
ходить із кількох потоків. Без checkpointer-а HITL не працює взагалі: між `interrupt()` і
`resume` стан має десь жити.

«Падіння» демонструється чесно, двома **окремими процесами**:

```
процес 1 (--demo persistence-start):  читає стрім вузлів і обриває його після першого act
процес 2 (--demo persistence-resume): app.get_state(config) показує відновлені plan/results,
                                      app.invoke(None, config) доводить план до кінця
```

`thread_id` = сесія: різні `thread_id` — незалежні ланцюжки виконання (`--demo threads`).

## LLM-провайдер

`ChatOpenAI` з LangChain, спрямований на **OpenRouter** (`https://openrouter.ai/api/v1`),
модель `xiaomi/mimo-v2.5`, embeddings `qwen/qwen3-embedding-8b`. Усе — константи в `config.py`.
Structured outputs через `with_structured_output(..., method="function_calling")`.

Ключ — `OPEN_ROUTER_API_KEY` у `.env` (gitignored).

## Запуск

```bash
cd 04_plan-and-execute-with-memory-and-hitl
./setup.sh                     # .venv + залежності + шаблон .env
./run.sh                       # усі демонстрації → demo_results.json + agent_state.db

# або поодинці:
./run.sh --demo plan                 # Завдання 1: planner/executor/replanner
./run.sh --demo rag                  # Завдання 3: RAG vs інші інструменти
./run.sh --demo hitl                 # Завдання 4: approve / reject / edit
./run.sh --demo persistence-start    # Завдання 2: «падіння» посередині плану
./run.sh --demo persistence-resume   # ...і продовження в новому процесі
./run.sh --demo threads              # незалежність thread_id
./run.sh --query "..." --thread my-1 # довільний запит

./tear_down.sh                 # прибрати .venv, chroma_db та артефакти

source .venv/bin/activate && pytest    # 50 тестів, офлайн, без витрат на API
```

### Живий лог

Кожен вузол пише, що робить, з таймстемпом від старту процесу (`plan_agent/logs.py`) —
без цього агент десятками секунд молчить у термінал і не зрозуміло, чи він працює, чи
чекає на 429 від провайдера:

```
  [   4.8s] planner   │ ціль: Дізнатися вартість 50 євро в гривнях та туристичний збір у Празі
  [   4.8s] planner   │   1. currency(amount=50, code='EUR') — отримати курс євро
  [   4.8s] executor  │ крок 1/2: currency(amount=50, code='EUR') — отримати курс євро
  [   6.7s] executor  │ обрано currency({"amount": 50, "code": "EUR"})
  [   7.0s] act       │ currency → {'amount': 50.0, 'rate': 51.0693, 'uah': 2553.47, …}
  [  11.7s] replanner │ continue — Другий крок плану досі актуальний для досягнення мети
  [  14.3s] knowledge │ RAG-запит: tourist tax Prague (топ-3)
  [  26.2s] replanner │ finish — план виконано повністю
  [  26.2s] respond   │ збираю фінальну відповідь із 2 результатів
```

Ризиковий інструмент видно окремо: `executor │ обрано book_hotel({…}) ⚠ РИЗИКОВИЙ`, далі
`approval │ ⚠ book_hotel(…) — чекаю рішення людини (interrupt)` і `approval │ APPROVE` /
`REJECT` / `APPROVE (аргументи змінено людиною)`.

Артефакти здачі: `agent_state.db` (стан усіх демо-сесій), `demo_results.json` (план,
результати кроків, рішення replanner-а, використані інструменти, payload interrupt-ів,
відповіді людини).

## Аналіз результатів

Прогін `./run.sh` від 2026-07-28, модель `xiaomi/mimo-v2.5`, `max_iterations=8`.

| Демо | Кроків плану | Інструментів | Час | Рішення replanner-а | Результат |
|---|---|---|---|---|---|
| `plan_simple` — 200 EUR + безвіз | 2/2 | 2 | 24.3 с | continue, finish | ✅ `search_knowledge` + `currency` |
| `plan_complex` — 3 ночі у Кракові | 4/4 | 3 | 67.3 с | continue×3, finish | ✅ geocode → weather → RAG → підсумок |
| `rag_reference` — скасування + страховка | 2/2 | 2 | 56.5 с | continue, finish | ✅ лише `search_knowledge`, живих API не торкнувся |
| `rag_live` — погода Львова + 100 USD | 3/3 | 3 | 19.9 с | continue×2, finish | ✅ geocode → weather → currency, RAG **не** викликано |
| `hitl` approve | 2/2 | 2 | — | continue, finish | ✅ interrupt → approve → `WE-KRA-20260810` |
| `hitl` reject | 3/3 | 3 | — | continue×2, finish | ✅ interrupt → reject → бронювання не виконано |
| `hitl` edit | 3/3 | 3 | — | continue×2, finish | ✅ виконано з параметрами людини (`WE-КРА-20260811`) |
| `persistence` | 1/4 → 4/4 | 4 | — | — | ✅ «падіння» після кроку 1, новий процес довів до кінця |
| `threads` | 2/2 | 2 | — | continue, finish | ✅ свіжий thread порожній, `persist-001` не змінився |

**Plan-and-Execute працює за задумом.** Planner тримає розумну гранулярність (2–4 кроки) і
не планує того, чого не просили: у `plan_complex` він сам не поставив бронювання, а фінальна
відповідь містила `actions: ["Бронювання не здійснювалося (користувач не просив)"]`. Ланцюжок
`geocode → weather` жодного разу не порушено, хоча правило живе лише в системному промпті.

**Agentic RAG вибирає джерело правильно.** Той самий агент на довідковому запиті зробив два
виклики `search_knowledge` і **не пішов** у живі API; на запиті про погоду й курс —
`geocode`/`weather`/`currency` і **жодного** RAG. Тобто маршрутизацію робить модель за
docstring-ами, а не наш код.

**Persistence.** Процес 1 пройшов `planner → executor → approval → act` і обірвався:
в `agent_state.db` лишився крок 1/4 і `next=('replanner',)`. Процес 2 — інший інтерпретатор,
інше з'єднання з файлом — прочитав ту саму ціль і результат першого кроку та довів план до
4/4. `thread_id` ізольовані: свіжий thread бачив `values={}`, а чужий прогін не змінив
`persist-001` (як до, так і після — крок 4, та сама ціль).

**HITL.** Усі три сценарії відпрацювали на одному й тому самому графі, різниця лише у
`Command(resume=...)`. Найцікавіше — `edit`: людина підмінила дату, кількість ночей і суму,
`act` виконав бронювання саме з її аргументами, а `respond` сам помітив розбіжність із
початковим запитом і виніс її у `warnings` («заїзд 2026-08-11 (а не 10-го), 1 ніч (а не 2)»).
На `reject` агент не впав і не спробував обійти відмову: крок отримав результат «ВІДХИЛЕНО»,
replanner пішов далі, фінальна відповідь чесно каже, що бронювання немає.

**Де модель помиляється.**
- `rag_live`: замість `currency(100, USD)` викликала `currency(1, USD)` і у відповіді дала
  курс за 1 долар, не помноживши на 100 — питання користувача було про 100. Тобто планування
  правильне, а перенесення числа з запиту в аргумент — ні.
- Назву міста нормалізує непослідовно: у `hitl-approve` пішло `city='Krakow'`, у
  `hitl-reject` — `'Kraków'`, в `hitl-edit` — `'Краків'`. Для `book_hotel` це видно в
  референсі бронювання (`WE-KRA-…` проти `WE-КРА-…`).
- Останнім кроком планів часто ставить «сформувати відповідь». Інструментів там немає, тож
  крок закривається текстом — і цей текст дублює роботу вузла `respond`, ще й у Markdown з
  таблицями та емодзі, хоча схема просить коротко.
- У кроки плану вписує сигнатури інструментів (`geocode('Krakow') — отримати координати`).
  Шкоди немає, executor однаково сам обирає інструмент, але план перестає бути читабельним
  для людини.

**Replanning у живих прогонах не спрацював жодного разу** — модель у всіх шести прогонах
обирала `continue`, поки план не вичерпувався, а тоді `finish`. Це очікувано для коротких
планів без несподіванок: у наших кейсах жоден інструмент не повернув помилки, яка вимагала
б переписати хвіст. Гілка `replan` відтворюється тестом
`tests/test_graph.py::test_replan_replaces_only_remaining_steps`, де фейкова модель віддає
`ReplanDecision(action="replan", ...)` і перевіряється, що виконані кроки збереглися, а
решта замінена.

**Запобіжник `max_iterations`** у живих прогонах теж не спрацював (плани були коротші за
ліміт) — перевіряється тестом `test_max_iterations_guard_stops_endless_replanning`.

## Тестування

50 офлайн-тестів, жодного звернення до мережі, до платного API чи до справжньої ChromaDB:

- `tests/test_tools.py` — валідація всіх схем (порожнє місто, `days=8`, `code="US"`,
  `nights=0/31`, `check_in="10.08.2026"`, `total_cost=0`), розбір відповідей API,
  перетворення мережевої помилки на зрозуміле повідомлення.
- `tests/test_knowledge.py` — база знань має ≥ 8 змістовних документів, `search_knowledge`
  збирає топ-N, поважає `n_results`, віддає `error` на порожній базі та на зламаному Chroma.
- `tests/test_graph.py` — покроковий прогін плану, `replan` переписує лише хвіст плану,
  `finish` не виконує решту кроків, запобіжник `max_iterations`, HITL (interrupt зупиняє
  перед виконанням; approve виконує **без повторного виклику LLM**; reject не виконує; edit
  бере аргументи людини; `resume=True`; невалідні аргументи стають результатом кроку),
  persistence (новий граф + нове з'єднання до того самого файлу відновлює стан і доводить
  план до кінця) та незалежність `thread_id`.

Мокається межа: `requests.get` для інструментів, об'єкт LLM для графа, `collection`/`embed`
для RAG. Тверджень про конкретні формулювання моделі немає — лише про структуру.

## Інженерні рішення

**Чому `replan` не скидає `current_step` у нуль.** У прикладі з методички `replan` ставить
`current_step: 0`, тобто агент переробляє вже виконані кроки — зайві виклики API і ризик
подвійного бронювання. Тут `plan = plan[:current_step] + updated_steps`: історія
недоторкана, переписується лише те, що ще не зроблено.

**Чому `pending`/`decisions` живуть у стані.** Це єдиний спосіб передати вибір LLM у вузол
підтвердження й далі у виконання, не викликаючи модель повторно після `resume`.

**Чому `log` окремо від `messages`.** `messages` — людиночитний журнал для промптів;
`log` — плаский машинний слід (`plan`/`action`/`observation`/`approval`/`rejected`/
`decision`/`reasoning`), з якого збираються артефакти й будуються перевірки в демо
(«чи справді викликано `search_knowledge`», «чи справді `book_hotel` не виконано»).

**Чому відмова людини не зупиняє граф.** Reject — це нормальний результат кроку, а не збій:
`act` пише «ВІДХИЛЕНО» у `results`, `replanner` вирішує, що робити далі, `respond` чесно
згадує це в `actions`/`warnings`.
