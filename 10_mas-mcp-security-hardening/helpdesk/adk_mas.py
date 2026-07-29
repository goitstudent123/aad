"""Бонус: третя реалізація у Google ADK — workflow agent (SequentialAgent) над тими ж MCP tools."""

import asyncio
import json
import time

from .config import LITELLM_MODEL, RISKY_TOOLS, litellm_env
from .guardrails import GuardrailError, check_tool_call, detect_injection, redact_pii
from .mcp_client import load_mcp_tools
from .prompts import DOMAIN, WORKERS
from .tracing import langfuse_span

_TOOLS: dict = {}


async def _call(agent: str, name: str, args: dict) -> str:
    try:
        clean = check_tool_call(agent, name, args)
    except GuardrailError as error:
        return f"ВІДХИЛЕНО guardrail-ом — {error}"
    if name in RISKY_TOOLS:
        return "ВІДМОВЛЕНО: у цій реалізації незворотні дії вимкнено (HITL живе у LangGraph)."
    return json.dumps(await _TOOLS[name].ainvoke(clean), ensure_ascii=False, default=str)


async def search_kb(query: str) -> str:
    """Шукає статті бази знань за ключовими словами."""
    return await _call("kb_agent", "search_kb", {"query": query})


async def lookup_employee(employee_id: str) -> str:
    """Картка співробітника за табельним номером (формат E1001)."""
    return await _call("directory_agent", "lookup_employee", {"employee_id": employee_id})


async def list_tickets(employee_id: str, status: str = "open") -> str:
    """Заявки співробітника: open, closed або all."""
    return await _call(
        "directory_agent", "list_tickets", {"employee_id": employee_id, "status": status}
    )


def build_adk_agent():
    """Два кроки послідовного workflow: збір фактів → відповідь користувачу."""
    from google.adk.agents import LlmAgent, SequentialAgent
    from google.adk.models.lite_llm import LiteLlm

    litellm_env()
    model = LiteLlm(model=LITELLM_MODEL)

    researcher = LlmAgent(
        name="researcher",
        model=model,
        instruction=f"{WORKERS['kb_agent']}\n{WORKERS['directory_agent']}\n"
                    "Збери факти інструментами й перелічи їх стисло.",
        tools=[search_kb, lookup_employee, list_tickets],
        output_key="facts",
    )
    writer = LlmAgent(
        name="writer",
        model=model,
        instruction=f"{DOMAIN}\nЗібрані факти: {{facts}}\n"
                    "Дай відповідь українською лише за цими фактами.",
    )
    return SequentialAgent(name="nexora_workflow", sub_agents=[researcher, writer])


def run_adk(query: str) -> dict:
    """Прогін ADK-агента з тими самими guardrails на вході й виході."""
    from google.adk.runners import InMemoryRunner

    verdict = detect_injection(query)
    if verdict.blocked:
        return {"query": query, "answer": verdict.reason, "blocked": verdict.reason}

    _TOOLS.update({tool.name: tool for tool in asyncio.run(load_mcp_tools())})
    runner = InMemoryRunner(agent=build_adk_agent(), app_name="nexora-adk")

    started = time.monotonic()
    with langfuse_span("adk.run", query=query):
        events = asyncio.run(runner.run_debug(query, quiet=True))
    elapsed = round(time.monotonic() - started, 2)

    texts = [
        part.text
        for event in events
        if event.content and event.content.parts
        for part in event.content.parts
        if getattr(part, "text", None)
    ]
    usages = [e.usage_metadata for e in events if getattr(e, "usage_metadata", None)]
    answer, found = redact_pii(texts[-1] if texts else "(порожня відповідь)")
    return {
        "query": query,
        "answer": answer,
        "pii_redacted": found,
        "usage": {
            "llm_calls": len(usages),
            "input_tokens": sum(u.prompt_token_count or 0 for u in usages),
            "output_tokens": sum(u.candidates_token_count or 0 for u in usages),
            "seconds": elapsed,
        },
    }
