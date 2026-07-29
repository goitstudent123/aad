"""Точка входу: одна демонстрація (--demo) або довільний запит (--query)."""

import argparse

from . import demos
from .config import DB_PATH, MAX_ITERATIONS, make_saver
from .graph import build_graph, initial_state, summarise
from .trajectory import merge_json

RESULTS_FILE = "demo_results.json"

DEMOS = {
    "plan": demos.demo_plan,
    "rag": demos.demo_rag,
    "hitl": demos.demo_hitl,
    "persistence-start": demos.demo_persistence_start,
    "persistence-resume": demos.demo_persistence_resume,
    "threads": demos.demo_threads,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan-and-Execute агент з memory та HITL")
    parser.add_argument("--demo", choices=[*DEMOS, "all"], help="яку демонстрацію запустити")
    parser.add_argument("--query", help="довільний запит замість демонстрації")
    parser.add_argument("--thread", default="cli-001", help="thread_id для --query")
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    args = parser.parse_args()

    # SqliteSaver тримаємо відкритим на весь час роботи процесу.
    saver = make_saver(DB_PATH)
    app = build_graph(checkpointer=saver)

    if args.query:
        config = {"configurable": {"thread_id": args.thread}}
        state = app.invoke(initial_state(args.query, args.max_iterations), config)
        result = summarise(state)
        demos.print_state(result)
        if result["interrupt"]:
            print(f"\n    граф зупинено на interrupt: {result['interrupt']}")
            print("    підтвердити: --demo hitl або власний Command(resume=...)")
        return

    if not args.demo:
        parser.error("вкажи --demo або --query")

    names = [args.demo] if args.demo != "all" else ["plan", "rag", "hitl", "threads"]
    for name in names:
        merge_json(RESULTS_FILE, name, DEMOS[name](app))
    print(f"\nРезультати → {RESULTS_FILE}, стан → {DB_PATH}")


if __name__ == "__main__":
    main()
