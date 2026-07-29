"""Бонус: той самий кейс у CrewAI — hierarchical manager + ті самі 3 ролі й MCP-інструменти.

Guardrails спільні з LangGraph-реалізацією: вхід перевіряється до kickoff, інструменти —
у тій самій обгортці, вихід — тим самим redact. HITL у CrewAI немає: замість паузи графа
тут блокуючий callback, за замовчуванням — відмова (див. README, розділ порівняння).
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional

from pydantic import BaseModel

from config import (
    ALLOWLIST,
    API_KEY_ENV,
    BASE_URL,
    LITELLM_MODEL,
    PRICE_PER_MTOK,
    RISKY_TOOLS,
    ROOT,
    litellm_env,
)
from guardrails import ARG_SCHEMAS, GuardrailError, check_tool_call, input_guardrail, output_guardrail
from logs import short, trace
from mcp_client import all_tools
from prompts import DOMAIN, WORKERS

ROLES = {
    "billing": ("Білінг-агент", "Розбирати платежі, нарахування та повернення коштів"),
    "tech": ("Технічний агент", "Діагностувати збої пристроїв і давати кроки виправлення"),
    "researcher": ("Довідковий агент", "Відповідати з бази знань про правила та строки"),
}

# Скільки коду тримає ту саму систему в кожному фреймворку.
IMPLEMENTATIONS = {
    "LangGraph": ["mas_langgraph.py", "plan_execute.py", "react_core.py", "mcp_client.py"],
    "CrewAI": ["mas_crewai.py"],
}


def _guarded_crew_tool(source_tool, agent: str, approve: Optional[Callable]):
    """Той самий guardrail-конвеєр, але у вигляді CrewAI BaseTool."""
    from crewai.tools import BaseTool

    class GuardedTool(BaseTool):
        name: str = source_tool.name
        description: str = source_tool.description
        args_schema: type[BaseModel] = ARG_SCHEMAS[source_tool.name]
        agent_name: str = agent
        approver: Optional[Any] = approve

        def _run(self, **kwargs):
            try:
                args = check_tool_call(self.agent_name, self.name, kwargs)
            except GuardrailError as error:
                return f"ВІДХИЛЕНО guardrail-ом — {error}"
            if self.name in RISKY_TOOLS and not (self.approver and self.approver(self.name, args)):
                return "ВІДМОВЛЕНО людиною."
            result = asyncio.run(source_tool.ainvoke(args))
            return json.dumps(result, ensure_ascii=False, default=str)

    return GuardedTool()


def build_crew(tools: list, approve: Optional[Callable] = None):
    """Crew із трьох агентів під керівництвом manager_llm — аналог супервізора LangGraph."""
    from crewai import LLM, Agent, Crew, Process, Task

    litellm_env()
    # Без цього CrewAI питає про телеметрію інтерактивно й вішає неінтерактивний запуск.
    os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
    llm = LLM(model=LITELLM_MODEL, api_key=os.environ[API_KEY_ENV], base_url=BASE_URL,
              temperature=0.0)
    by_name = {tool.name: tool for tool in tools}

    agents = [
        Agent(
            role=role,
            goal=goal,
            backstory=WORKERS[name],
            tools=[_guarded_crew_tool(by_name[tool_name], name, approve)
                   for tool_name in sorted(ALLOWLIST[name]) if tool_name in by_name],
            llm=llm,
            # Делегування вимкнене: інакше виконавчі агенти ходять по колу.
            allow_delegation=False,
            verbose=False,
        )
        for name, (role, goal) in ROLES.items()
    ]

    task = Task(
        description=(f"{DOMAIN}\nЗвернення клієнта: {{query}}\n"
                     "Збери факти інструментами й дай відповідь українською. "
                     "Незворотні дії — лише після підтвердження людини."),
        expected_output="Коротка відповідь українською: що з'ясовано, що робити далі, що виконано.",
    )
    return Crew(agents=agents, tasks=[task], process=Process.hierarchical,
                manager_llm=llm, verbose=False)


def run_crew(query: str, approve: Optional[Callable] = None) -> dict:
    """Прогін CrewAI з тими самими guardrails: вхід, інструменти, вихід."""
    safe, cleaned = input_guardrail(query)
    if not safe:
        return {"query": query, "answer": cleaned, "blocked": cleaned,
                "usage": {"llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "seconds": 0}}

    tools = asyncio.run(all_tools())
    crew = build_crew(tools, approve=approve)

    started = time.monotonic()
    result = crew.kickoff(inputs={"query": cleaned})
    elapsed = round(time.monotonic() - started, 2)

    answer, found = output_guardrail(str(result))
    usage = getattr(result, "token_usage", None) or getattr(crew, "usage_metrics", None)
    usage = usage.model_dump() if hasattr(usage, "model_dump") else (usage or {})
    trace("crewai", f"готово за {elapsed}s, PII приховано: {found}")
    return {
        "query": query,
        "answer": answer,
        "pii_redacted": found,
        "usage": {
            "llm_calls": usage.get("successful_requests", 0),
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "seconds": elapsed,
        },
    }


# ── Порівняння фреймворків ─────────────────────────────────────────────────


def cost_usd(usage: dict) -> float:
    input_price, output_price = PRICE_PER_MTOK
    return round(usage.get("input_tokens", 0) / 1e6 * input_price
                 + usage.get("output_tokens", 0) / 1e6 * output_price, 6)


def loc(paths: list[str]) -> int:
    """Рядки без порожніх і без коментарів."""
    total = 0
    for path in paths:
        for line in Path(ROOT / path).read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                total += 1
    return total


def loc_by_framework() -> dict:
    return {name: loc(paths) for name, paths in IMPLEMENTATIONS.items()}


def compare_table(usages: dict, locs: dict) -> list[dict]:
    return [
        {
            "framework": name,
            "loc": locs.get(name, 0),
            "llm_calls": usage.get("llm_calls", 0),
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "seconds": usage.get("seconds", 0),
            "cost_usd": cost_usd(usage),
        }
        for name, usage in usages.items()
    ]


if __name__ == "__main__":
    print(short(run_crew("Не списано платіж за тариф у вересні, тікет TKT-001, клієнт C-100"), 800))
