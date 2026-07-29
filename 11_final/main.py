"""Точка входу: демонстрації (--demo) або довільний запит (--query)."""

import argparse
import asyncio
import json
from pathlib import Path

import demos
from config import RESULTS_FILE, make_saver
from hitl import resume_command
from logs import trace
from mas_langgraph import build_default, print_result, result_of, run_mas
from observability import SpanRecorder, run_config, save_trace
from trajectory_logger import save_trajectory


def merge_json(path, key: str, value) -> None:
    """Демо запускаються окремими процесами — файл результатів дописуємо."""
    file = Path(path)
    data = json.loads(file.read_text(encoding="utf-8")) if file.exists() else {}
    data[key] = value
    file.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


async def one_query(query: str, thread: str, decision: str | None) -> dict:
    """Довільний запит. Якщо в потоці висить пауза — рішення береться з --approve/--reject."""
    recorder = SpanRecorder(thread)
    config = run_config(thread, recorder)
    async with make_saver() as saver:
        app = await build_default(checkpointer=saver)
        snapshot = await app.aget_state(config)
        if snapshot.tasks and decision:
            trace("main", f"продовжую паузу в потоці {thread}: {decision}")
            state = await app.ainvoke(resume_command(decision),
                                      {**config, "recursion_limit": 40})
            result = result_of(state)
        else:
            result = await run_mas(app, query, config)
        print_result(result)
        if result["pending_approval"]:
            print(f"\n  Дія чекає рішення: повтори запуск із --approve або --reject "
                  f"і тим самим --thread {thread}")
    save_trajectory(result["trajectory"], key=thread)
    save_trace({thread: recorder})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="MAS служби підтримки: MCP + guardrails + HITL")
    parser.add_argument("--demo", choices=[*demos.DEMOS, "all"])
    parser.add_argument("--query", help="довільне звернення користувача")
    parser.add_argument("--thread", default="cli-001", help="thread_id для HITL-паузи")
    parser.add_argument("--approve", action="store_true", help="підтвердити ризикову дію")
    parser.add_argument("--reject", action="store_true", help="відхилити ризикову дію")
    args = parser.parse_args()

    if args.query is not None:
        decision = "approve" if args.approve else "reject" if args.reject else None
        asyncio.run(one_query(args.query, args.thread, decision))
        return

    names = demos.ORDER if args.demo in (None, "all") else [args.demo]
    for name in names:
        print(f"\n{'═' * 78}\n▶ демо: {name}\n{'═' * 78}")
        merge_json(RESULTS_FILE, name, asyncio.run(demos.DEMOS[name]()))
    print(f"\nРезультати демонстрацій → {RESULTS_FILE.name}")


if __name__ == "__main__":
    main()
