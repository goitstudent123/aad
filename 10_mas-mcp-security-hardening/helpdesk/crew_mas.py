"""Той самий кейс у CrewAI: manager (hierarchical) + ті самі 3 ролі та ті самі MCP-інструменти."""

import asyncio
import json
import os
import time
from typing import Any, Callable, Optional

from pydantic import BaseModel

from .config import ALLOWLIST, API_KEY_ENV, BASE_URL, LITELLM_MODEL, RISKY_TOOLS, litellm_env
from .guardrails import ARG_SCHEMAS, GuardrailError, check_tool_call, detect_injection, redact_pii
from .mcp_client import load_mcp_tools
from .prompts import DOMAIN, WORKERS
from .tracing import langfuse_span

ROLES = {
    "kb_agent": ("Довідковий агент", "Знаходити інструкції та політики в базі знань"),
    "directory_agent": ("Агент каталогу", "Знаходити картку співробітника та його заявки"),
    "ops_agent": ("Виконавчий агент", "Виконувати незворотні дії лише з дозволу людини"),
}


def _guarded_crew_tool(mcp_tool, agent: str, approve: Optional[Callable]):
    from crewai.tools import BaseTool

    class GuardedTool(BaseTool):
        name: str = mcp_tool.name
        description: str = mcp_tool.description
        args_schema: type[BaseModel] = ARG_SCHEMAS[mcp_tool.name]
        agent_name: str = agent
        approver: Optional[Any] = approve

        def _run(self, **kwargs):
            try:
                args = check_tool_call(self.agent_name, self.name, kwargs)
            except GuardrailError as error:
                return f"ВІДХИЛЕНО guardrail-ом — {error}"
            if self.name in RISKY_TOOLS and not (self.approver and self.approver(self.name, args)):
                return "ВІДМОВЛЕНО людиною."
            result = asyncio.run(mcp_tool.ainvoke(args))
            return json.dumps(result, ensure_ascii=False, default=str)

    return GuardedTool()


def build_crew(mcp_tools: list, approve: Optional[Callable] = None):
    """Crew із трьох агентів під керівництвом manager_llm — аналог супервізора LangGraph."""
    from crewai import LLM, Agent, Crew, Process, Task

    litellm_env()
    # Без цього CrewAI питає про телеметрію інтерактивно й вішає неінтерактивний запуск.
    os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
    llm = LLM(
        model=LITELLM_MODEL,
        api_key=os.environ[API_KEY_ENV],
        base_url=BASE_URL,
        temperature=0.0,
    )
    by_name = {tool.name: tool for tool in mcp_tools}

    agents = [
        Agent(
            role=role,
            goal=goal,
            backstory=WORKERS[name],
            tools=[
                _guarded_crew_tool(by_name[tool_name], name, approve)
                for tool_name in sorted(ALLOWLIST[name])
                if tool_name in by_name
            ],
            llm=llm,
            allow_delegation=False,
            verbose=False,
        )
        for name, (role, goal) in ROLES.items()
    ]

    task = Task(
        description=(
            f"{DOMAIN}\nЗвернення користувача: {{query}}\n"
            "Збери факти інструментами й дай відповідь українською. "
            "Незворотні дії — лише після підтвердження людини."
        ),
        expected_output="Коротка відповідь українською: що з'ясовано, що зробити далі, що виконано.",
    )

    return Crew(
        agents=agents,
        tasks=[task],
        process=Process.hierarchical,
        manager_llm=llm,
        verbose=False,
    )


def run_crew(query: str, approve: Optional[Callable] = None) -> dict:
    """Прогін CrewAI з тими самими guardrails: вхід, інструменти, вихід."""
    verdict = detect_injection(query)
    if verdict.blocked:
        return {"query": query, "answer": verdict.reason, "blocked": verdict.reason}

    mcp_tools = asyncio.run(load_mcp_tools())
    crew = build_crew(mcp_tools, approve=approve)

    started = time.monotonic()
    with langfuse_span("crewai.kickoff", query=query):
        result = crew.kickoff(inputs={"query": query})
    elapsed = round(time.monotonic() - started, 2)

    answer, found = redact_pii(str(result))
    usage = getattr(result, "token_usage", None) or getattr(crew, "usage_metrics", None)
    usage = usage.model_dump() if hasattr(usage, "model_dump") else (usage or {})
    return {
        "query": query,
        "answer": answer,
        "pii_redacted": found,
        "seconds": elapsed,
        "usage": {
            "llm_calls": usage.get("successful_requests", 0),
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "seconds": elapsed,
        },
    }
