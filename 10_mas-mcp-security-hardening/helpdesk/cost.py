"""Бонус: cost tracking — токени, час і гроші по фреймворках, плюс LOC для порівняння."""

from pathlib import Path

from .config import PRICE_PER_MTOK, ROOT

# Скільки коду тримає ту саму систему в кожному фреймворку.
IMPLEMENTATIONS = {
    "LangGraph": ["helpdesk/graph.py", "helpdesk/mcp_client.py"],
    "CrewAI": ["helpdesk/crew_mas.py"],
    "ADK": ["helpdesk/adk_mas.py"],
}


def cost_usd(usage: dict) -> float:
    input_price, output_price = PRICE_PER_MTOK
    return round(
        usage.get("input_tokens", 0) / 1e6 * input_price
        + usage.get("output_tokens", 0) / 1e6 * output_price,
        6,
    )


def loc(paths: list[str]) -> int:
    """Рядки без порожніх і без коментарів."""
    total = 0
    for path in paths:
        for line in Path(ROOT / path).read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                total += 1
    return total


def compare(usages: dict) -> list[dict]:
    """usages: {'LangGraph': {...}, 'CrewAI': {...}} → рядки порівняльної таблиці."""
    return [
        {
            "framework": name,
            "loc": loc(IMPLEMENTATIONS.get(name, [])),
            "llm_calls": usage.get("llm_calls", 0),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "seconds": usage.get("seconds", 0),
            "cost_usd": cost_usd(usage),
        }
        for name, usage in usages.items()
    ]


def as_table(rows: list[dict]) -> str:
    header = "| Фреймворк | LOC | LLM-викликів | Токени in/out | Час, с | Вартість, $ |"
    separator = "|---|---|---|---|---|---|"
    lines = [
        f"| {r['framework']} | {r['loc']} | {r['llm_calls']} | "
        f"{r['input_tokens']}/{r['output_tokens']} | {r['seconds']} | {r['cost_usd']:.6f} |"
        for r in rows
    ]
    return "\n".join([header, separator, *lines])
