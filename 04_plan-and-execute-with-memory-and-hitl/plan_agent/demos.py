"""Демонстрації всіх чотирьох завдань: план, persistence, agentic RAG, HITL.

Кожна демонстрація друкує хід роботи й повертає словник, який main.py складає в
demo_results.json. Демо persistence розбите на два процеси (start/resume) — інакше
«перезапуск» був би несправжнім.
"""

import time

from langgraph.types import Command

from .graph import initial_state, summarise

QUERIES = {
    "plan_simple": "Скільки гривень коштує 200 євро і що потрібно українцю для безвізу до ЄС?",
    "plan_complex": (
        "Плануємо три ночі у Кракові в готелі 3*: яка там буде погода, "
        "скільки орієнтовно коштує житло і що з туристичним збором?"
    ),
    "rag_reference": "Чи можна безкоштовно скасувати бронювання готелю і що зі страховкою для Шенгену?",
    "rag_live": "Яка зараз погода у Львові і скільки гривень коштує 100 доларів?",
    "hitl_booking": (
        "Забронюй готель Rynek Inn у Кракові із заїздом 2026-08-10 на 2 ночі за 180 EUR."
    ),
    "persistence": (
        "Готую вихідні у Кракові: перевір погоду, правила безвізу та переведи 200 EUR у гривні."
    ),
}


def _tools_used(state: dict) -> list[str]:
    return [entry["signature"] for entry in state.get("log", []) if entry["type"] == "action"]


def print_state(state: dict) -> None:
    print(f"    ціль: {state.get('goal')}")
    for i, step in enumerate(state.get("plan", []), start=1):
        mark = "✓" if i <= state.get("current_step", 0) else " "
        print(f"    [{mark}] {i}. {step}")
    for result in state.get("results", []):
        print(f"      → {result[:300]}")
    if state.get("stop_reason"):
        print(f"    ЗУПИНЕНО: {state['stop_reason']}")
    answer = state.get("answer") or {}
    if answer:
        print(f"    відповідь: {answer.get('summary')}")
        for fact in answer.get("facts", []):
            print(f"      • {fact}")
        for action in answer.get("actions", []):
            print(f"      ⚑ {action}")
        for warning in answer.get("warnings", []):
            print(f"      ! {warning}")


def _reset(app, thread: str) -> None:
    """Чистить checkpoint-и цього thread_id.

    Демо мають бути повторюваними: якщо не чистити, повторний запуск дописує кроки й лог
    у стару сесію, і артефакт показує суміш двох прогонів.
    """
    app.checkpointer.delete_thread(thread)


def _invoke(app, payload, config, thread: str | None = None, attempts: int = 3):
    """app.invoke із повтором: OpenRouter періодично віддає 429.

    thread задають для НОВОГО прогону — тоді перед кожною спробою сесія чиститься, щоб
    напіввиконаний стан не змішувався з наступною спробою. Для resume чистити не можна:
    саме у стані лежить те, що ми продовжуємо.
    """
    for attempt in range(1, attempts + 1):
        if thread:
            _reset(app, thread)
        try:
            return app.invoke(payload, config)
        except Exception as exc:  # noqa: BLE001 — провайдер може впасти будь-як
            print(f"    спроба {attempt}/{attempts} впала: {exc}", flush=True)
            if attempt == attempts:
                raise
            print(f"    чекаю {10 * attempt}с і пробую ще раз", flush=True)
            time.sleep(10 * attempt)


def _run(app, query: str, thread: str) -> dict:
    """Повний прогін одного запиту з нуля."""
    config = {"configurable": {"thread_id": thread}}
    started = time.monotonic()
    try:
        state = _invoke(app, initial_state(query), config, thread=thread)
    except Exception as exc:  # noqa: BLE001 — демо не має валити решту прогону
        return {"query": query, "thread_id": thread, "error": str(exc)}
    result = summarise(state)
    result |= {
        "query": query,
        "thread_id": thread,
        "tools_used": _tools_used(state),
        "elapsed_seconds": round(time.monotonic() - started, 2),
    }
    return result


# ── Завдання 1: Plan-and-Execute ──────────────────────────────────────────────
def demo_plan(app) -> list[dict]:
    results = []
    for name in ("plan_simple", "plan_complex"):
        print(f"\n=== [plan] {name}: {QUERIES[name]}")
        result = _run(app, QUERIES[name], f"plan-{name}")
        print_state(result)
        print(f"    інструменти: {result.get('tools_used')}")
        print(f"    рішення replanner-а: "
              f"{[e['action'] for e in result.get('log', []) if e['type'] == 'decision']}")
        results.append({"name": name, **result})
    return results


# ── Завдання 3: Agentic RAG ───────────────────────────────────────────────────
def demo_rag(app) -> list[dict]:
    """Той самий агент, два різні запити: агент сам вибирає базу знань або живі API."""
    results = []
    for name, expected in (("rag_reference", "має піти в search_knowledge"),
                           ("rag_live", "має піти в geocode/weather/currency, без RAG")):
        print(f"\n=== [rag] {name}: {QUERIES[name]}")
        print(f"    очікуємо: {expected}")
        result = _run(app, QUERIES[name], f"rag-{name}")
        print_state(result)
        used = result.get("tools_used", [])
        print(f"    інструменти: {used}")
        print(f"    search_knowledge викликано: "
              f"{any(s.startswith('search_knowledge') for s in used)}")
        results.append({"name": name, "expected": expected, **result})
    return results


# ── Завдання 4: HITL ──────────────────────────────────────────────────────────
def _run_until_interrupt(app, query: str, thread: str) -> tuple[dict, dict]:
    config = {"configurable": {"thread_id": thread}}
    state = _invoke(app, initial_state(query), config, thread=thread)
    payload = state["__interrupt__"][0].value if state.get("__interrupt__") else {}
    return state, payload


def _hitl_scenario(app, thread: str, reply: dict, title: str) -> dict:
    print(f"\n=== [hitl] {title} (thread {thread})")
    state, payload = _run_until_interrupt(app, QUERIES["hitl_booking"], thread)
    if not payload:
        print("    interrupt НЕ спрацював — модель не викликала ризиковий інструмент")
        return {"name": title, "thread_id": thread, "interrupted": False, **summarise(state)}

    print("    ГРАФ ЗУПИНЕНО на interrupt(), деталі дії:")
    print(f"      tool: {payload.get('tool')}")
    print(f"      args: {payload.get('args')}")
    config = {"configurable": {"thread_id": thread}}
    print(f"    наступний вузол за checkpoint-ом: {app.get_state(config).next}")
    print(f"    відповідь людини: {reply}")

    # Ризикових викликів у кроці може бути кілька — тоді буде кілька interrupt-ів.
    while state.get("__interrupt__"):
        state = _invoke(app, Command(resume=reply), config)

    result = summarise(state)
    executed = [e for e in result["log"] if e["type"] == "observation" and e["tool"] == "book_hotel"]
    rejected = [e for e in result["log"] if e["type"] == "rejected"]
    print(f"    book_hotel виконано: {bool(executed)}; відхилено: {bool(rejected)}")
    print_state(result)
    return {"name": title, "thread_id": thread, "interrupted": True, "human_reply": reply,
            "booked": bool(executed), "rejected": bool(rejected),
            "tools_used": _tools_used(result), "interrupt_payload": payload,
            "query": QUERIES["hitl_booking"], **result}


def demo_hitl(app) -> list[dict]:
    return [
        _hitl_scenario(app, "hitl-approve", {"approved": True}, "approve — бронювання виконується"),
        _hitl_scenario(app, "hitl-reject", {"approved": False, "reason": "занадто дорого"},
                       "reject — бронювання НЕ виконується"),
        _hitl_scenario(
            app,
            "hitl-edit",
            {"approved": True, "args": {"hotel_name": "Rynek Inn", "city": "Краків",
                                        "check_in": "2026-08-11", "nights": 1, "total_cost": 95.0}},
            "edit — людина змінює параметри перед виконанням",
        ),
    ]


# ── Завдання 2: persistence ───────────────────────────────────────────────────
def demo_persistence_start(app, thread: str = "persist-001") -> dict:
    """Перший процес: доходимо до першого виконаного кроку і «падаємо»."""
    config = {"configurable": {"thread_id": thread}}
    print(f"\n=== [persistence] процес 1: старт (thread {thread})")
    _reset(app, thread)  # щоб «падіння» щоразу відбувалося з чистого аркуша
    visited = []
    for event in app.stream(initial_state(QUERIES["persistence"]), config, stream_mode="updates"):
        node = next(iter(event))
        visited.append(node)
        print(f"    вузол: {node}")
        if node == "act":
            # Імітація падіння процесу: просто перестаємо читати стрім і виходимо.
            print("    ⛔ імітуємо аварійне завершення процесу після першого кроку")
            break

    state = app.get_state(config)
    print(f"    у checkpoint-і: крок {state.values.get('current_step')}"
          f"/{len(state.values.get('plan', []))}, наступний вузол {state.next}")
    return {"thread_id": thread, "visited_nodes": visited, "next": list(state.next),
            "plan": state.values.get("plan"), "results": state.values.get("results"),
            "current_step": state.values.get("current_step")}


def demo_persistence_resume(app, thread: str = "persist-001") -> dict:
    """Другий процес: читаємо стан з agent_state.db і доводимо план до кінця."""
    config = {"configurable": {"thread_id": thread}}
    print(f"\n=== [persistence] процес 2: відновлення (thread {thread})")
    restored = app.get_state(config)
    if not restored.values:
        print("    у базі немає стану для цього thread_id — спершу запусти persistence-start")
        return {"thread_id": thread, "restored": False}

    print(f"    відновлено з agent_state.db: ціль '{restored.values.get('goal')}'")
    print(f"    крок {restored.values.get('current_step')}/{len(restored.values.get('plan', []))}, "
          f"наступний вузол {restored.next}")
    for result in restored.values.get("results", []):
        print(f"      → {result[:200]}")

    # invoke(None) = «продовжуй з останнього checkpoint-а», без нового вхідного стану.
    state = _invoke(app, None, config)
    print("    план доведено до кінця у новому процесі:")
    result = summarise(state)
    print_state(result)
    return {"thread_id": thread, "restored": True,
            "restored_step": restored.values.get("current_step"),
            "restored_results": restored.values.get("results"),
            "restored_next": list(restored.next), **result}


def demo_threads(app, thread: str = "thread-independent") -> dict:
    """Незалежність thread_id: свіжий thread не бачить нічого з persist-001."""
    print("\n=== [threads] незалежність сесій")
    other = app.get_state({"configurable": {"thread_id": "persist-001"}})
    fresh = app.get_state({"configurable": {"thread_id": thread}})
    print(f"    persist-001: кроків {other.values.get('current_step')}, "
          f"ціль '{other.values.get('goal')}'")
    print(f"    {thread} до запуску: values={fresh.values or '{}'}")

    result = _run(app, QUERIES["plan_simple"], thread)
    print_state(result)
    after_other = app.get_state({"configurable": {"thread_id": "persist-001"}})
    print(f"    persist-001 після чужого прогону не змінився: "
          f"кроків {after_other.values.get('current_step')}, ціль '{after_other.values.get('goal')}'")
    return {
        "persist_001_before": {"goal": other.values.get("goal"),
                               "current_step": other.values.get("current_step")},
        "fresh_before_empty": not fresh.values,
        "fresh_run": {"goal": result.get("goal"), "plan": result.get("plan"),
                      "tools_used": result.get("tools_used")},
        "persist_001_after": {"goal": after_other.values.get("goal"),
                              "current_step": after_other.values.get("current_step")},
    }
