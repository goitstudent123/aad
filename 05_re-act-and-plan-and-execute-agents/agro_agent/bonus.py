"""Додаткові вимоги: порівняння ReAct vs Plan-and-Execute, порівняння моделей,
mermaid-візуалізація графів, async-версія на AsyncSqliteSaver та fallback-стратегія.
"""

import asyncio
import json

from . import tools
from .config import PROVIDERS, make_llm, make_saver
from .demos import QUERIES, print_state, retry
from .plan_execute import build_plan_graph, initial_state, run_plan, summarise
from .react import build_react_graph, run_react
from .tools import call_tool

ASYNC_DB_PATH = "agent_state_async.db"
GRAPHS_FILE = "graphs.md"

COMPARE_QUERY = QUERIES["plan_complex"]


def _facts(result: dict) -> int:
    answer = result.get("answer") or {}
    return len(answer.get("facts", []))


def _summary(result: dict) -> str:
    return ((result.get("answer") or {}).get("summary") or "")


# ── Бонус: ReAct vs Plan-and-Execute на одній задачі ──────────────────────────
def compare_agents(react_app=None, plan_app=None, **_) -> dict:
    print(f"\n=== [compare] одна задача двома патернами: {COMPARE_QUERY}")
    react_app = react_app or build_react_graph()
    plan_app = plan_app or build_plan_graph(checkpointer=make_saver())

    print("\n--- ReAct")
    react = retry(lambda: run_react(COMPARE_QUERY, graph=react_app))
    print(f"    кроків: {react['steps']}, інструментів: {len(react['tool_calls'])}, "
          f"час: {react['elapsed_seconds']}с")

    print("\n--- Plan-and-Execute")
    thread = "compare-001"
    config = {"configurable": {"thread_id": thread}}
    plan = retry(
        lambda: run_plan(plan_app, COMPARE_QUERY, config),
        reset=lambda: plan_app.checkpointer.delete_thread(thread),
    )
    print_state(plan)

    decisions = [e["action"] for e in plan["log"] if e.get("type") == "decision"]
    # Виклики LLM: planner + кроки вкладених ReAct-агентів + рішення replanner-а + respond.
    plan_llm_calls = 1 + plan["react_steps"] + len(decisions) + 1
    plan_tools = [e["signature"] for e in plan["log"] if e.get("type") == "action"]

    table = [
        ("викликів LLM", react["steps"] + 1, plan_llm_calls),
        ("викликів інструментів", len(react["tool_calls"]), len(plan_tools)),
        ("час, с", react["elapsed_seconds"], plan["elapsed_seconds"]),
        ("кроків плану", "—", len(plan["plan"])),
        ("фактів у відповіді", _facts(react), _facts(plan)),
        ("довжина відповіді", len(_summary(react)), len(_summary(plan))),
    ]
    print(f"\n    {'метрика':24} {'ReAct':>12} {'Plan-and-Execute':>18}")
    for metric, left, right in table:
        print(f"    {metric:24} {str(left):>12} {str(right):>18}")

    return {
        "query": COMPARE_QUERY,
        "react": {
            "llm_calls": react["steps"] + 1,
            "tool_calls": react["tool_calls"],
            "elapsed_seconds": react["elapsed_seconds"],
            "facts": _facts(react),
            "answer": react.get("answer"),
        },
        "plan_and_execute": {
            "llm_calls": plan_llm_calls,
            "plan": plan["plan"],
            "replanner_decisions": decisions,
            "tool_calls": plan_tools,
            "elapsed_seconds": plan["elapsed_seconds"],
            "facts": _facts(plan),
            "answer": plan.get("answer"),
        },
        "table": [{"metric": m, "react": r, "plan_and_execute": p} for m, r, p in table],
    }


# ── Бонус: порівняння LLM-моделей на одному запиті ─────────────────────────────
def compare_providers(**_) -> dict:
    query = QUERIES["react_chain"]
    print(f"\n=== [providers] один запит на {len(PROVIDERS)} моделях: {query}")
    rows = []
    for model in PROVIDERS:
        print(f"\n--- {model}")
        try:
            graph = build_react_graph(make_llm(model=model))
            result = retry(lambda: run_react(query, graph=graph), attempts=2)
            rows.append({
                "model": model,
                "llm_calls": result["steps"] + 1,
                "tool_calls": result["tool_calls"],
                "elapsed_seconds": result["elapsed_seconds"],
                "facts": _facts(result),
                "stop_reason": result.get("stop_reason"),
                "summary": _summary(result),
            })
            print(f"    кроків: {result['steps']}, час: {result['elapsed_seconds']}с, "
                  f"інструменти: {result['tool_calls']}")
            print(f"    відповідь: {_summary(result)[:300]}")
        except Exception as exc:  # noqa: BLE001 — модель може бути недоступна
            print(f"    модель недоступна: {exc}")
            rows.append({"model": model, "error": str(exc)})

    print(f"\n    {'модель':32} {'LLM':>5} {'інстр.':>7} {'час, с':>8} {'фактів':>7}")
    for row in rows:
        print(f"    {row['model']:32} {row.get('llm_calls', '—'):>5} "
              f"{len(row.get('tool_calls', [])):>7} {row.get('elapsed_seconds', '—'):>8} "
              f"{row.get('facts', '—'):>7}")
    return {"query": query, "rows": rows}


# ── Бонус: візуалізація графів ────────────────────────────────────────────────
def draw_graphs(react_app=None, plan_app=None, **_) -> dict:
    react_app = react_app or build_react_graph()
    plan_app = plan_app or build_plan_graph(checkpointer=make_saver())
    react_mermaid = react_app.get_graph().draw_mermaid()
    plan_mermaid = plan_app.get_graph().draw_mermaid()
    with open(GRAPHS_FILE, "w", encoding="utf-8") as file:
        file.write("# Графи агентів\n\n## ReAct\n\n```mermaid\n")
        file.write(react_mermaid)
        file.write("```\n\n## Plan-and-Execute\n\n```mermaid\n")
        file.write(plan_mermaid)
        file.write("```\n")
    print(f"\n=== [graph] mermaid-схеми → {GRAPHS_FILE}")
    print(plan_mermaid)
    return {"file": GRAPHS_FILE, "react": react_mermaid, "plan_and_execute": plan_mermaid}


# ── Бонус: async-версія (AsyncSqliteSaver + ainvoke) ──────────────────────────
async def _run_async(query: str) -> dict:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    async with AsyncSqliteSaver.from_conn_string(ASYNC_DB_PATH) as saver:
        app = build_plan_graph(checkpointer=saver)
        config = {"configurable": {"thread_id": "async-001"}}
        # На порожньому файлі таблиць ще немає, тому спершу setup, потім чистка сесії.
        await saver.setup()
        await saver.adelete_thread("async-001")
        state = await app.ainvoke(initial_state(query), config)
        snapshot = await app.aget_state(config)
        result = summarise(state)
        result["next"] = list(snapshot.next)
        return result


def demo_async(**_) -> dict:
    query = QUERIES["rag_norms"]
    print(f"\n=== [async] той самий граф через ainvoke та AsyncSqliteSaver: {query}")
    result = asyncio.run(_run_async(query))
    print_state(result)
    print(f"    стан збережено в {ASYNC_DB_PATH}, наступний вузол: {result['next']}")
    return {"query": query, "db": ASYNC_DB_PATH, **result}


# ── Бонус: fallback-стратегія на відмову інструмента ──────────────────────────
def demo_fallback(**_) -> dict:
    """Основні джерела навмисно ламаються — call_tool має піти в резервні."""
    print("\n=== [fallback] основне джерело недоступне → автоматична альтернатива")
    broken = "https://127.0.0.1:9/dead"
    original = (tools.GEOCODE_URL, tools.NBU_URL)
    results = {}
    try:
        tools.GEOCODE_URL, tools.NBU_URL = broken, broken
        for name, args in (("locate_field", {"settlement": "Умань"}),
                           ("input_cost", {"amount": 240, "code": "USD"})):
            output = json.loads(call_tool(name, args))
            print(f"    {name}: status={output['status']}, "
                  f"джерело={output.get('data', {}).get('source')}")
            results[name] = output
    finally:
        tools.GEOCODE_URL, tools.NBU_URL = original

    healthy = json.loads(call_tool("locate_field", {"settlement": "Умань"}))
    print(f"    після відновлення основного джерела: {healthy.get('data', {}).get('source')}")
    results["locate_field_healthy"] = healthy
    return results
