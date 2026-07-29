"""Точка входу: демонстрації (--demo) або довільний запит (--query)."""

import argparse

from . import bonus, demos
from .config import DB_PATH, MAX_ITERATIONS, MAX_STEPS, TIMEOUT_SECONDS, make_llm, make_saver
from .plan_execute import build_plan_graph, run_plan
from .react import build_react_graph, run_react
from .trajectory import merge_json

RESULTS_FILE = "demo_results.json"
TRAJECTORY_FILE = "trajectory.json"

DEMOS = {
    "react": lambda react_app, plan_app: demos.demo_react(react_app),
    "plan": lambda react_app, plan_app: demos.demo_plan(plan_app),
    "rag": lambda react_app, plan_app: demos.demo_rag(react_app),
    "hitl": lambda react_app, plan_app: demos.demo_hitl(plan_app),
    "persistence-start": lambda react_app, plan_app: demos.demo_persistence_start(plan_app),
    "persistence-resume": lambda react_app, plan_app: demos.demo_persistence_resume(plan_app),
    "memory": lambda react_app, plan_app: demos.demo_memory(plan_app),
    "compare": lambda react_app, plan_app: bonus.compare_agents(react_app, plan_app),
    "providers": lambda react_app, plan_app: bonus.compare_providers(),
    "graph": lambda react_app, plan_app: bonus.draw_graphs(react_app, plan_app),
    "async": lambda react_app, plan_app: bonus.demo_async(),
    "fallback": lambda react_app, plan_app: bonus.demo_fallback(),
}

ALL = [
    "graph", "react", "plan", "rag", "hitl", "memory",
    "compare", "providers", "async", "fallback",
]


def _trajectories(payload) -> list[dict]:
    """Витягує з результату демо все, що є траєкторією виконання."""
    items = payload if isinstance(payload, list) else [payload]
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        steps = item.get("trajectory") or item.get("log")
        if steps:
            out.append({"name": item.get("name") or item.get("query"), "steps": steps})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="ReAct та Plan-and-Execute агенти агронома")
    parser.add_argument("--demo", choices=[*DEMOS, "all"], help="яку демонстрацію запустити")
    parser.add_argument("--query", help="довільний запит замість демонстрації")
    parser.add_argument("--agent", choices=["react", "plan"], default="plan",
                        help="яким агентом виконувати --query")
    parser.add_argument("--thread", default="cli-001", help="thread_id для --query")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--timeout", type=float, default=TIMEOUT_SECONDS)
    parser.add_argument("--max-iterations", type=int, default=MAX_ITERATIONS)
    args = parser.parse_args()

    llm = make_llm()
    react_app = build_react_graph(llm)

    if args.query and args.agent == "react":
        result = run_react(args.query, args.max_steps, args.timeout, graph=react_app)
        demos.print_react(result)
        merge_json(TRAJECTORY_FILE, "cli", _trajectories(result))
        return

    # SqliteSaver тримаємо відкритим на весь час роботи процесу.
    saver = make_saver(DB_PATH)
    plan_app = build_plan_graph(llm, checkpointer=saver, react_graph=react_app)

    if args.query:
        config = {"configurable": {"thread_id": args.thread}}
        result = run_plan(plan_app, args.query, config, args.max_iterations)
        demos.print_state(result)
        if result["pending"]:
            print(f"\n    граф зупинено перед ризиковою дією: "
                  f"{[c['name'] for c in result['pending']]}")
            print("    підтвердити: --demo hitl або власний update_state({'approval': ...})")
        merge_json(TRAJECTORY_FILE, "cli", _trajectories(result))
        return

    if not args.demo:
        parser.error("вкажи --demo або --query")

    for name in ([args.demo] if args.demo != "all" else ALL):
        payload = DEMOS[name](react_app, plan_app)
        merge_json(RESULTS_FILE, name, payload)
        trajectories = _trajectories(payload)
        if trajectories:
            merge_json(TRAJECTORY_FILE, name, trajectories)
    print(f"\nРезультати → {RESULTS_FILE}, траєкторії → {TRAJECTORY_FILE}, стан → {DB_PATH}")


if __name__ == "__main__":
    main()
