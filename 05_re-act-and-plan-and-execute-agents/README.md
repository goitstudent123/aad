# Практична робота №5 — ReAct та Plan-and-Execute агенти

Агентна система **«Поле під контролем»** для нової предметної області — агрономії
середнього господарства. Агент радить, коли сіяти, обприскувати, підживлювати й поливати:
норми бере з бази знань, живі показники поля — з відкритих API, а незворотні дії
(наряд на обробку поля пестицидом) виконує лише після підтвердження людини.

Реалізовані **обидва** патерни на одних і тих самих інструментах:

- **ReAct** — цикл «думка → інструмент → спостереження», рішення про наступний крок
  ухвалюється після кожного результату.
- **Plan-and-Execute** — повний план складається заздалегідь, кожен крок плану виконує
  **вкладений ReAct-агент**, а після кроку `replanner` вирішує `continue / replan / finish`.

## Швидкий старт

```bash
cd 05_re-act-and-plan-and-execute-agents
./setup.sh                      # .venv + залежності + шаблон .env
# впиши OPEN_ROUTER_API_KEY у .env
./run.sh                        # усі демонстрації по черзі
./tear_down.sh                  # прибрати .venv, chroma_db, agent_state.db, артефакти
```

Окремі запуски:

```bash
./run.sh --demo react           # ReAct + захисні механізми
./run.sh --demo plan            # Plan-and-Execute
./run.sh --demo rag             # agentic RAG: агент сам вибирає базу знань чи живі дані
./run.sh --demo hitl            # human-in-the-loop: approve і reject
./run.sh --demo persistence-start && ./run.sh --demo persistence-resume
./run.sh --demo memory          # стан між запитами в одному thread_id
./run.sh --demo compare         # бонус: ReAct vs Plan-and-Execute у числах
./run.sh --demo providers       # бонус: порівняння моделей
./run.sh --demo graph           # бонус: mermaid-схеми графів → graphs.md
./run.sh --demo async           # бонус: AsyncSqliteSaver + ainvoke
./run.sh --demo fallback        # бонус: резервне джерело при відмові основного

./run.sh --query "Чи можна обприскувати поле біля Умані завтра?" --agent react
./run.sh --query "Порахуй норму азоту для 40 га" --agent plan --thread my-001
```

Тести (офлайн, без ключа й мережі):

```bash
source .venv/bin/activate
pytest -v
pytest tests/test_plan_execute.py::test_reject_does_not_execute_the_tool -v
```

## Структура

```
agro_agent/
  config.py         # модель, embedding-модель, шляхи до agent_state.db і chroma_db, ліміти
  prompts.py        # системний промпт домену, спільний для обох агентів
  schemas.py        # structured outputs: Plan, ReplanDecision, AgroAnswer
  tools.py          # 5 інструментів + call_tool із fallback-стратегією + RISKY_TOOLS
  knowledge.py      # ChromaDB (11 документів) + інструмент search_knowledge
  react.py          # ReAct-граф і захисні механізми
  plan_execute.py   # Plan-and-Execute граф: planner → executor → risky_act → replanner
  demos.py          # демонстрації обов'язкових вимог
  bonus.py          # порівняння агентів і моделей, mermaid, async, fallback
  logs.py           # живий трейс ходу роботи з таймстемпами
  trajectory.py     # траєкторія виконання та запис артефактів
  main.py           # CLI
tests/              # 69 офлайн-тестів (LLM, мережа та Chroma замокані)
```

## Інструменти (Pydantic v2)

| Інструмент | Джерело | Схема, валідація |
|---|---|---|
| `locate_field(settlement)` | Open-Meteo Geocoding, резерв — OSM Nominatim | `FieldArgs`: непорожня назва, не лише цифри, ≤100 символів |
| `soil_forecast(latitude, longitude, days)` | Open-Meteo Forecast | `SoilArgs`: широта −90…90, довгота −180…180, `days` 1…7 |
| `input_cost(amount, code)` | НБУ, резерв — exchangerate-api | `CostArgs`: `amount > 0`, ISO-код із трьох літер, не UAH |
| `search_knowledge(query, n_results)` | ChromaDB | `SearchArgs`: непорожній запит, `n_results` 1…5 |
| `schedule_spraying(...)` ⚠ **ризиковий** | наряд бригади | `SprayingArgs`: площа 0…5000 га, норма 0…20 л/га, дата `YYYY-MM-DD` |

Кожен інструмент повертає **JSON-рядок** зі стандартним конвертом:

```json
{"status": "ok", "data": {"soil_temperature_6cm_c": 16.8, "…": "…"}}
{"status": "error", "error": "джерело … недоступне: connection refused"}
```

Помилка мережі чи валідації не валить граф — вона стає спостереженням, і агент бачить причину.

## ReAct-агент і захисні механізми

```
START → agent ──(tool_calls)──► tools ──► agent ──► … ──► respond → END
```

| Механізм | Де | Що робить |
|---|---|---|
| `max_steps=10` | `agent_node` | ліміт ітерацій циклу, далі — `stop_reason=max_steps` |
| `timeout=120 с` | `agent_node` | загальний дедлайн прогону, далі — `stop_reason=timeout` |
| детекція повторів | `agent_node` | той самий інструмент з тими самими аргументами → `stop_reason=loop_detected` |
| `recursion_limit` | компіляція | підстраховка LangGraph, якщо власні лічильники підведуть |

Зупинка не втрачає роботу: `respond` збирає відповідь із зібраного й перелічує в `warnings`,
чого не встиг дізнатися (видно у демо `guard_max_steps` і `guard_timeout`).

**Ризиковий виклик ReAct-цикл не виконує.** Побачивши `schedule_spraying`, вузол `agent`
кладе виклик у `pending` і виходить із циклу — виконання чекає на рішення людини у
Plan-and-Execute графі.

## Plan-and-Execute

```
START → planner → executor ──(ризикова дія)──► risky_act ──┐
                     │                                     │
                     └────────(звичайний крок)────────────►replanner
                                                              │
                             ┌────────────(не завершено)──────┘
                             ▼
                         executor        (завершено) ──► respond ──► END
```

| Вузол | Що робить |
|---|---|
| `planner` | `with_structured_output(Plan)` → `goal` + 2–5 кроків одним викликом |
| `executor` | виконує ОДИН крок плану **вкладеним ReAct-агентом** (`run_react`) |
| `risky_act` | виконує ризиковий виклик — лише якщо у стані лежить підтвердження людини |
| `replanner` | `with_structured_output(ReplanDecision)` → `continue` / `replan` / `finish` |
| `respond` | `with_structured_output(AgroAnswer)`: `summary`, `facts[]`, `actions[]`, `warnings[]` |

`replan` переписує **лише хвіст** плану: виконані кроки лишаються на місці.
Запобіжник `max_iterations=8` не дає циклу `replan → executor → replan` крутитися вічно.
Вкладений ReAct обмежений своїми `STEP_MAX_STEPS=4` і `STEP_TIMEOUT_SECONDS=60`.

**Чому виконання ризикової дії — окремий вузол.** `interrupt_before` зупиняє граф перед
вузлом, а після `resume` вузол виконується з початку. Якби вибір інструментів і його
виконання жили в одному вузлі, після кожного «approve» модель обирала б аргументи вдруге —
і могла б підставити не ті, які підтвердила людина. Тому інструменти обирає `executor`,
а виконує `risky_act`, який до моделі не звертається (тест
`test_approve_executes_the_tool_without_reselecting_arguments`).

## Checkpointer і відновлення

`SqliteSaver` на файлі `agent_state.db` (не `:memory:` — інакше стан не переживе процес).

- **Стан між запитами** — демо `memory`: два запити в одному `thread_id`, другий бачить
  контекст першого; сторонній `thread_id` не бачить нічого.
- **Відновлення після переривання** — демо `persistence-start` / `persistence-resume`, це
  два **окремі процеси**. Перший доходить до ризикової дії й завершується, другий відкриває
  той самий файл і доводить план до кінця.
- **Знімок стану** — `app.get_state(config)`: `values` (план, крок, `pending`) і `next`
  (вузол, перед яким граф стоїть).

## Agentic RAG

ChromaDB, **11 документів** агротехнічних норм: строки й умови сівби, умови обприскування,
змивання дощем, періоди очікування, норми азоту, вологість ґрунту, розрахунок поливу,
ЕПШ шкідників, стійкість до заморозків, зберігання ЗЗР. Embeddings — `qwen/qwen3-embedding-8b`
через OpenRouter, колекція persistent (не перераховується щозапуску).

Рішення «база знань чи живі дані» ухвалює **сама модель** за описами інструментів: у базі
свідомо немає живих показників, а `soil_forecast`/`input_cost` не знають норм. Демо `rag`
показує обидва шляхи на двох запитах — і в логу видно, який інструмент агент вибрав.

## Human-in-the-Loop

`interrupt_before=["risky_act"]` при компіляції (лише разом із checkpointer — без нього
паузу нічим відновити). Далі рішення людини кладеться у стан і прогін продовжується:

```python
app.invoke(initial_state(query), config)              # стоп перед risky_act
app.get_state(config).next                            # ('risky_act',)
app.update_state(config, {"approval": {"approved": True}})
app.invoke(None, config)                              # продовження з того ж чекпойнта
```

- **approve** → інструмент виконується з підтвердженими аргументами;
  `{"approved": true, "args": {...}}` дозволяє виконати із **зміненими** параметрами.
- **reject** → інструмент не виконується взагалі, у результат кроку йде «ВІДХИЛЕНО» з
  причиною, агент завершує роботу нормальною відповіддю.
- **немає рішення** → трактується як відмова: незворотну дію без явного «так» не виконуємо.

## Артефакти

| Файл | Що там |
|---|---|
| `trajectory.json` | JSON-лог траєкторій: `query` / `thought` / `action` / `observation` по кожному демо |
| `demo_results.json` | результати всіх демонстрацій, включно з таблицями порівнянь |
| `graphs.md` | mermaid-схеми обох графів |
| `agent_state.db` | checkpoint-и SqliteSaver |
| `agent_state_async.db` | checkpoint-и AsyncSqliteSaver з async-демо |
| `test_results.txt` | вивід `pytest -v` |

## Бонусні вимоги

- **ReAct vs Plan-and-Execute на одній задачі** (`--demo compare`). Запит:
  «Поле 3 біля Тернополя, 40 га: чи годиться погода й ґрунт для обробки гербіцидом цього
  тижня, і скільки коштуватиме препарат за 18 доларів за літр при нормі 2 л/га?»

  | Метрика | ReAct | Plan-and-Execute |
  |---|---|---|
  | викликів LLM | 4 | 14 |
  | викликів інструментів | 4 | 4 |
  | час, с | 57.7 | 200.2 |
  | кроків плану | — | 5 |
  | фактів у відповіді | 10 | 7 |
  | довжина відповіді, символів | 390 | 445 |

  Обидва агенти дійшли однакового висновку тими самими чотирма інструментами, але
  Plan-and-Execute витратив утричі більше часу й у 3.5 раза більше викликів моделі: за
  кожен крок платить окремий вкладений ReAct-агент плюс planner, replanner і respond.
  Виграш плану — не швидкість, а керованість: видно, які кроки заплановані, який з них
  провалився, і саме там вбудований шлюз підтвердження ризикової дії. На задачі з
  чотирьох очевидних викликів дешевший ReAct; план виправдовує себе там, де потрібні
  адаптація до результату, HITL і відновлення перерваного прогону.
  Числа одного прогону — у `demo_results.json → compare`.
- **Порівняння моделей** (`--demo providers`) — той самий запит («Чи можна сьогодні сіяти
  кукурудзу на полі біля Тернополя?») на двох моделях через OpenRouter:

  | Модель | викликів LLM | інструментів | час, с | фактів |
  |---|---|---|---|---|
  | `deepseek/deepseek-v4-flash` | 5 | 4 | 39.7 | 0 |
  | `google/gemini-2.5-flash-lite` | 5 | 3 | 5.8 | 3 |

  Обидві правильно вибудували ланцюг `locate_field → soil_forecast → search_knowledge`.
  Gemini відповіла у 7 разів швидше й трьома інструментами, але коротко — і не помітила,
  що 29 липня для сівби кукурудзи це безнадійно пізній строк. Deepseek зробила
  додатковий запит до бази знань саме про пізні строки й попередила про це, зате не
  заповнила `facts` у структурованій відповіді (усі числа лишилися в `summary`).
- **Візуалізація графа** (`--demo graph`) — `graph.get_graph().draw_mermaid()` → `graphs.md`.
- **Async-версія** (`--demo async`) — `AsyncSqliteSaver.from_conn_string` + `ainvoke` +
  `aget_state` на окремому файлі стану.
- **Fallback-стратегія** (`--demo fallback`) — `call_tool` бачить `status=error` від
  основного джерела й повторює запит через резервне: Open-Meteo → OSM Nominatim,
  НБУ → exchangerate-api. У відповіді видно поле `source`, тож підміна не є непомітною.
  Помилку валідації Pydantic резервне джерело не отримує — невалідні аргументи воно не
  врятує, і агент має побачити справжню причину.

## Тести

`pytest` — 69 офлайн-тестів, LLM і мережа замокані (`tests/conftest.py`).

| Файл | Що перевіряє |
|---|---|
| `test_schemas.py` | 36 тестів валідації: некоректні дати, площі, дози, коди валют, координати, `n_results`; нормалізація входів; `Plan`/`ReplanDecision` |
| `test_tools.py` | конверт `{status, data\|error}`, розбір відповідей API, підрахунок препарату, `call_tool` із fallback і без нього |
| `test_react.py` | цикл LLM–tools–LLM, траєкторія, `max_steps`, `timeout`, детекція повторів, ризиковий виклик не виконується |
| `test_plan_execute.py` | план крок за кроком, `replan` хвоста, `finish`, `max_iterations`, HITL approve/reject/edit/«без рішення», persistence між процесами, ізоляція `thread_id` |

## Модель

Усі звернення — через OpenRouter (`config.py`). Основна модель — `deepseek/deepseek-v4-flash`,
embeddings — `qwen/qwen3-embedding-8b`. Друга модель, `google/gemini-2.5-flash-lite`, потрібна
лише для бонусного порівняння провайдерів (`--demo providers`). Імена моделей живуть в одному
місці — `agro_agent/config.py`.
