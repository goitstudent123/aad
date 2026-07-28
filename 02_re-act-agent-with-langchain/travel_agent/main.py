"""Точка входу: один запит (--query) або повний прогін тест-кейсів (за замовчуванням)."""

import argparse
import time

from .cases import GUARD_DEMOS, TEST_CASES
from .config import MAX_STEPS, TIMEOUT_SECONDS
from .graph import build_graph, run_agent
from .trajectory import save_json

TRAJECTORY_FILE = "trajectory.json"
RESULTS_FILE = "test_results.json"


def _print(result: dict) -> None:
    answer = result.get("answer") or {}
    print(f"\n=== {result['query']}")
    print(f"    кроків: {result['steps']}, час: {result['elapsed_seconds']}с, "
          f"інструменти: {len(result['tool_calls'])}")
    if result.get("stop_reason"):
        print(f"    ЗУПИНЕНО: {result['stop_reason']}")
    print(f"    місто: {answer.get('city')}")
    print(f"    відповідь: {answer.get('summary')}")
    for fact in answer.get("facts", []):
        print(f"      • {fact}")
    for warning in answer.get("warnings", []):
        print(f"      ! {warning}")


def _run_case(graph, case: dict, attempts: int = 3) -> dict:
    """Прогін одного кейса. OpenRouter періодично віддає 429 — пробуємо ще раз,
    а якщо не вийшло, записуємо помилку і йдемо далі, щоб не втратити решту прогону."""
    for attempt in range(1, attempts + 1):
        try:
            result = run_agent(
                case["query"],
                max_steps=case.get("max_steps", MAX_STEPS),
                timeout=case.get("timeout", TIMEOUT_SECONDS),
                graph=graph,
            )
            _print(result)
            return {"name": case["name"], "expected": case["expected"], **result}
        except Exception as exc:  # noqa: BLE001 — провайдер може впасти будь-як
            print(f"    спроба {attempt}/{attempts} впала: {exc}", flush=True)
            if attempt < attempts:
                time.sleep(10 * attempt)
    return {"name": case["name"], "expected": case["expected"], "query": case["query"],
            "error": "усі спроби впали", "trajectory": []}


def main() -> None:
    parser = argparse.ArgumentParser(description="ReAct тревел-планер на LangGraph")
    parser.add_argument("--query", help="один запит замість повного набору тест-кейсів")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    parser.add_argument("--timeout", type=float, default=TIMEOUT_SECONDS)
    args = parser.parse_args()

    graph = build_graph()

    if args.query:
        result = run_agent(args.query, args.max_steps, args.timeout, graph=graph)
        _print(result)
        save_json(TRAJECTORY_FILE, [result["trajectory"]])
        print(f"\nТраєкторія → {TRAJECTORY_FILE}")
        return

    cases = [_run_case(graph, case) for case in TEST_CASES]
    guards = [_run_case(graph, case) for case in GUARD_DEMOS]

    save_json(RESULTS_FILE, {"test_cases": cases, "guard_demos": guards})
    save_json(TRAJECTORY_FILE, [r["trajectory"] for r in cases + guards])
    print(f"\nРезультати → {RESULTS_FILE}, траєкторії → {TRAJECTORY_FILE}")


if __name__ == "__main__":
    main()
