"""Точка входу: демонстрації (--demo) або довільний запит (--query)."""

import argparse
import asyncio

from . import demos
from .artifacts import merge_json
from .config import RESULTS_FILE, make_saver
from .graph import build_default, resume_mas, run_mas
from .tracing import SpanRecorder, export_langfuse_trace, run_config, save_trace

ASYNC_DEMOS = {
    "langgraph": demos.demo_langgraph,
    "hitl": demos.demo_hitl,
    "guardrails": demos.demo_guardrails,
}
SYNC_DEMOS = {
    "crew": demos.demo_crew,
    "adk": demos.demo_adk,
    "compare": demos.demo_compare,
    "redteam": demos.demo_redteam,
    "deepteam": demos.demo_deepteam,
    "evals": demos.demo_evals,
}
ALL = ["redteam", "langgraph", "guardrails", "hitl", "crew", "adk", "compare", "evals", "deepteam"]


def run_demo(name: str):
    if name in ASYNC_DEMOS:
        return asyncio.run(ASYNC_DEMOS[name]())
    return SYNC_DEMOS[name]()


async def one_query(query: str, thread: str, approve: bool | None) -> dict:
    """Довільний запит. Якщо агент став на ризиковій дії — рішення береться з --approve."""
    recorder = SpanRecorder(thread)
    config = run_config(thread, recorder)
    async with make_saver() as saver:
        app = await build_default(checkpointer=saver)
        snapshot = await app.aget_state(config)
        # Продовження паузи, збереженої попереднім запуском процесу.
        if snapshot.tasks and approve is not None:
            result = await resume_mas(app, config, approved=approve, comment="рішення з CLI")
        else:
            result = await run_mas(app, query, config)
        demos._print(result)

        if result["pending_approval"]:
            print("\n  Дія чекає рішення: повтори запуск із --approve або --reject "
                  f"і тим самим --thread {thread}")

    save_trace({thread: recorder})
    export_langfuse_trace()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="MAS служби підтримки: MCP + guardrails + HITL")
    parser.add_argument("--demo", choices=[*ASYNC_DEMOS, *SYNC_DEMOS, "all"])
    parser.add_argument("--query", help="довільне звернення користувача")
    parser.add_argument("--thread", default="cli-001", help="thread_id для HITL-паузи")
    parser.add_argument("--approve", action="store_true", help="підтвердити ризикову дію")
    parser.add_argument("--reject", action="store_true", help="відхилити ризикову дію")
    args = parser.parse_args()

    if args.query:
        approve = True if args.approve else (False if args.reject else None)
        asyncio.run(one_query(args.query, args.thread, approve))
        return

    if not args.demo:
        parser.error("вкажи --demo або --query")

    for name in ([args.demo] if args.demo != "all" else ALL):
        print(f"\n=== demo: {name} ===")
        merge_json(RESULTS_FILE, name, run_demo(name))
    exported = export_langfuse_trace()
    print(f"\nРезультати → {RESULTS_FILE.name}, трейс → trace.json"
          + (f", Langfuse → {exported}" if exported else ""))


if __name__ == "__main__":
    main()
