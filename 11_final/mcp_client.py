"""Підключення MCP-сервера до агентів: MultiServerMCPClient + guardrails + HITL."""

import os
import sys

from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from config import ALLOWLIST, MCP_SERVER, RISKY_TOOLS
from guardrails import GuardrailError, check_tool_call
from hitl import approval_gate
from logs import short, trace
from tools_legacy import LOCAL_TOOLS

SERVERS = {
    "support": {
        "command": sys.executable,
        "args": [str(MCP_SERVER)],
        "transport": "stdio",
        "env": {**os.environ, "FASTMCP_LOG_LEVEL": "ERROR"},
    }
}


async def load_mcp_tools() -> list:
    """Піднімає MCP-сервер по stdio і віддає його інструменти як LangChain tools."""
    tools = await MultiServerMCPClient(SERVERS).get_tools()
    trace("mcp", f"підключено {len(tools)} MCP-інструментів: {[t.name for t in tools]}")
    return tools


async def all_tools() -> list:
    """MCP-інструменти (дані CRM) + локальні інструменти з ДЗ1/ДЗ2."""
    return [*await load_mcp_tools(), *LOCAL_TOOLS]


def guarded(tool, agent: str) -> StructuredTool:
    """Обгортка: allowlist, валідація аргументів, пауза на підтвердження людини."""

    async def call(**kwargs):
        try:
            args = check_tool_call(agent, tool.name, kwargs)
        except GuardrailError as error:
            trace("guard", f"⛔ {error}")
            return f"ВІДХИЛЕНО guardrail-ом — {error}"

        if tool.name in RISKY_TOOLS:
            decision = approval_gate(tool.name, agent, args)
            if not decision.approved:
                return f"ВІДМОВЛЕНО людиною. {decision.comment}".strip()
            args = decision.args

        result = await tool.ainvoke(args)
        trace("tools", f"{tool.name} → {short(result, 160)}")
        if tool.name in RISKY_TOOLS and decision.action == "edit":
            # Інакше агент вирішує, що аргументи змінила система, і так і пише клієнту.
            return f"ЛЮДИНА ВИПРАВИЛА АРГУМЕНТИ на {args}. Результат: {result}"
        return result

    return StructuredTool.from_function(
        coroutine=call,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
    )


def tools_for(agent: str, tools: list) -> list:
    """Інструменти, дозволені конкретному агенту, вже під захистом."""
    return [guarded(tool, agent) for tool in tools if tool.name in ALLOWLIST.get(agent, set())]
