"""Демонстрації обов'язкових вимог: ReAct із захистом, план, RAG, HITL, persistence."""

import time

from .config import MAX_ITERATIONS
from .plan_execute import initial_state, run_plan, summarise
from .react import run_react

QUERIES = {
    "react_simple": "Скільки гривень коштує гербіцид за 240 доларів і за якого вітру ним можна працювати?",
    "react_chain": "Чи можна сьогодні сіяти кукурудзу на полі біля Тернополя?",
    "plan_complex": (
        "Поле 3 біля Тернополя, 40 га: чи годиться погода й ґрунт для обробки гербіцидом "
        "цього тижня, і скільки коштуватиме препарат за 18 доларів за літр при нормі 2 л/га?"
    ),
    "rag_norms": "За якої температури ґрунту сіють кукурудзу і який період очікування у гліфосатів?",
    "rag_live": "Яка зараз температура і вологість ґрунту біля Умані та скільки це 500 євро в гривнях?",
    "hitl_spraying": (
        "Умови ми вже перевірили. Призначай обробку: Поле 3 біля Умані, препарат Раундап, "
        "40 га, дата 2026-08-05, норма 2 л/га."
    ),
    "memory_first": "Наше поле біля Умані, 40 га кукурудзи. Яка там зараз вологість ґрунту?",
    "memory_second": "А скільки води на тиждень потрібно цьому полю за таким випаровуванням?",
}

GUARDS = [
    {
        "name": "guard_max_steps",
        "query": "Порівняй умови для сівби кукурудзи біля Тернополя й біля Умані за нормами бази знань.",
        "max_steps": 2,
        "expected": "stop_reason=max_steps, часткова відповідь із warnings",
    },
    {
        "name": "guard_timeout",
        "query": "Чи можна обприскувати поле біля Умані завтра?",
        "timeout": 5.0,
        "expected": "stop_reason=timeout, часткова відповідь із warnings",
    },
]


def print_react(result: dict) -> None:
    answer = result.get("answer") or {}
    print(f"    кроків: {result['steps']}, час: {result['elapsed_seconds']}с, "
          f"інструменти: {result['tool_calls']}")
    if result.get("stop_reason"):
        print(f"    ЗУПИНЕНО: {result['stop_reason']}")
    if result.get("pending"):
        print(f"    чекає підтвердження: {[c['name'] for c in result['pending']]}")
    print(f"    відповідь: {answer.get('summary')}")
    for fact in answer.get("facts", []):
        print(f"      • {fact}")
    for warning in answer.get("warnings", []):
        print(f"      ! {warning}")


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


def retry(action, attempts: int = 3, reset=None):
    """Повтор із паузою: OpenRouter періодично віддає 429.

    reset викликається перед кожною спробою нового прогону, щоб напіввиконаний стан не
    змішувався з наступною спробою. Для resume reset передавати не можна.
    """
    for attempt in range(1, attempts + 1):
        if reset:
            reset()
        try:
            return action()
        except Exception as exc:  # noqa: BLE001 — провайдер може впасти будь-як
            print(f"    спроба {attempt}/{attempts} впала: {exc}", flush=True)
            if attempt == attempts:
                raise
            time.sleep(10 * attempt)


def _thread(name: str) -> dict:
    return {"configurable": {"thread_id": name}}


def _reset(app, thread: str):
    return lambda: app.checkpointer.delete_thread(thread)


# ── ReAct-агент із захисними механізмами ──────────────────────────────────────
def demo_react(react_app, **_) -> list[dict]:
    results = []
    for name in ("react_simple", "react_chain"):
        print(f"\n=== [react] {name}: {QUERIES[name]}")
        result = retry(lambda: run_react(QUERIES[name], graph=react_app))
        print_react(result)
        results.append({"name": name, **result})

    for guard in GUARDS:
        print(f"\n=== [react] {guard['name']}: {guard['query']}")
        result = retry(
            lambda: run_react(
                guard["query"],
                max_steps=guard.get("max_steps", 10),
                timeout=guard.get("timeout", 120.0),
                graph=react_app,
            )
        )
        print_react(result)
        results.append({"name": guard["name"], "expected": guard["expected"], **result})
    return results


# ── Plan-and-Execute ──────────────────────────────────────────────────────────
def demo_plan(app, **_) -> dict:
    name = "plan_complex"
    print(f"\n=== [plan] {name}: {QUERIES[name]}")
    result = retry(
        lambda: run_plan(app, QUERIES[name], _thread(f"plan-{name}")),
        reset=_reset(app, f"plan-{name}"),
    )
    print_state(result)
    print(f"    інструменти: {result['tools_used']}")
    print(f"    рішення replanner-а: "
          f"{[e['action'] for e in result['log'] if e.get('type') == 'decision']}")
    return {"name": name, **result}


# ── Agentic RAG: агент сам вибирає базу знань або живі дані ────────────────────
def demo_rag(react_app, **_) -> list[dict]:
    results = []
    for name, expected in (("rag_norms", "має піти в search_knowledge"),
                           ("rag_live", "має піти в soil_forecast та input_cost")):
        print(f"\n=== [rag] {name} ({expected}): {QUERIES[name]}")
        result = retry(lambda: run_react(QUERIES[name], graph=react_app))
        print_react(result)
        used = {call.split("(")[0] for call in result["tool_calls"]}
        print(f"    вибір агента: {sorted(used)}")
        results.append({"name": name, "expected": expected, "tools": sorted(used), **result})
    return results


# ── Human-in-the-loop: approve і reject на ризиковому інструменті ──────────────
def demo_hitl(app, **_) -> list[dict]:
    results = []
    for name, decision in (("approve", {"approved": True}),
                           ("reject", {"approved": False, "reason": "по полю ще не пройшов дощ"})):
        thread = f"hitl-{name}"
        config = _thread(thread)
        print(f"\n=== [hitl:{name}] {QUERIES['hitl_spraying']}")

        retry(
            lambda: app.invoke(initial_state(QUERIES["hitl_spraying"]), config),
            reset=_reset(app, thread),
        )
        snapshot = app.get_state(config)
        pending = snapshot.values.get("pending", [])
        print(f"    граф стоїть перед: {snapshot.next}, чекає: {[c['name'] for c in pending]}")

        # Рішення людини кладемо у стан і продовжуємо з того ж чекпойнта.
        app.update_state(config, {"approval": decision})
        final = summarise(retry(lambda: app.invoke(None, config)))
        print(f"    рішення людини: {decision}")
        print_state(final)
        results.append({"name": name, "decision": decision,
                        "interrupted_before": list(snapshot.next), **final})
    return results


# ── Persistence: два окремих процеси на одному thread_id ──────────────────────
def demo_persistence_start(app, **_) -> dict:
    """Перший процес: доходить до ризикової дії, зупиняється і «падає»."""
    thread = "persist-001"
    config = _thread(thread)
    print(f"\n=== [persistence:start] {QUERIES['hitl_spraying']}")
    state = retry(
        lambda: app.invoke(initial_state(QUERIES["hitl_spraying"]), config),
        reset=_reset(app, thread),
    )
    snapshot = app.get_state(config)
    result = summarise(state)
    print(f"    зупинка перед: {snapshot.next}, виконано кроків: {result['current_step']}")
    print("    процес завершується — стан лишається у SQLite")
    return {"thread_id": thread, "next": list(snapshot.next), **result}


def demo_persistence_resume(app, **_) -> dict:
    """Другий процес: читає стан із файлу через get_state і доводить план до кінця."""
    thread = "persist-001"
    config = _thread(thread)
    snapshot = app.get_state(config)
    if not snapshot.values:
        print("\n=== [persistence:resume] стану немає — спершу запусти persistence-start")
        return {"thread_id": thread, "error": "стан не знайдено"}

    print("\n=== [persistence:resume] знімок стану з іншого процесу (get_state)")
    print(f"    ціль: {snapshot.values.get('goal')}")
    print(f"    план: {snapshot.values.get('plan')}")
    print(f"    виконано кроків: {snapshot.values.get('current_step')}")
    print(f"    наступний вузол: {snapshot.next}")
    print(f"    чекає підтвердження: {[c['name'] for c in snapshot.values.get('pending', [])]}")

    app.update_state(config, {"approval": {"approved": True}})
    final = summarise(retry(lambda: app.invoke(None, config)))
    print_state(final)
    return {"thread_id": thread, "restored_step": snapshot.values.get("current_step"),
            "restored_next": list(snapshot.next), **final}


# ── Пам'ять між запитами: той самий thread_id проти іншого ─────────────────────
def demo_memory(app, **_) -> dict:
    thread = "memory-001"
    config = _thread(thread)
    print(f"\n=== [memory] запит 1: {QUERIES['memory_first']}")
    retry(
        lambda: run_plan(app, QUERIES["memory_first"], config, max_iterations=MAX_ITERATIONS),
        reset=_reset(app, thread),
    )
    first = app.get_state(config)

    print(f"\n=== [memory] запит 2 у тому ж thread: {QUERIES['memory_second']}")
    second_result = retry(lambda: run_plan(app, QUERIES["memory_second"], config))
    print_state(second_result)

    other = app.get_state(_thread("memory-other"))
    print(f"    повідомлень у thread після двох запитів: "
          f"{len(app.get_state(config).values.get('messages', []))}")
    print(f"    чужий thread бачить: {other.values or 'нічого'}")
    return {
        "thread_id": thread,
        "messages_after_first": len(first.values.get("messages", [])),
        "messages_after_second": len(app.get_state(config).values.get("messages", [])),
        "other_thread_empty": not other.values,
        "second_answer": second_result.get("answer"),
    }
