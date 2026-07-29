"""Підключення MCP-сервера до агентів: langchain-mcp-adapters + guardrails + HITL."""

import os
import sys

from langchain_core.tools import StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.types import interrupt

from .config import ALLOWLIST, MCP_SERVER, RISKY_TOOLS
from .guardrails import GuardrailError, check_tool_call

SERVERS = {
    "helpdesk": {
        "command": sys.executable,
        "args": [str(MCP_SERVER)],
        "transport": "stdio",
        "env": {**os.environ, "FASTMCP_LOG_LEVEL": "ERROR"},
    }
}


async def load_mcp_tools() -> list:
    """Піднімає MCP-сервер по stdio і віддає його інструменти як LangChain tools."""
    return await MultiServerMCPClient(SERVERS).get_tools()


def guarded(tool, agent: str) -> StructuredTool:
    """Обгортка інструмента: allowlist, валідація аргументів і пауза на підтвердження людини."""

    async def call(**kwargs):
        try:
            args = check_tool_call(agent, tool.name, kwargs)
        except GuardrailError as error:
            return f"ВІДХИЛЕНО guardrail-ом — {error}"

        if tool.name in RISKY_TOOLS:
            decision = interrupt(
                {
                    "tool": tool.name,
                    "agent": agent,
                    "args": args,
                    "question": f"Дозволити незворотну дію {tool.name}({args})?",
                }
            )
            if not (decision if isinstance(decision, bool) else decision.get("approved")):
                comment = "" if isinstance(decision, bool) else decision.get("comment", "")
                return f"ВІДМОВЛЕНО людиною. {comment}".strip()

        return await tool.ainvoke(args)

    return StructuredTool.from_function(
        coroutine=call,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
    )


def tools_for(agent: str, tools: list) -> list:
    """Інструменти, дозволені конкретному агенту, вже під захистом."""
    return [guarded(tool, agent) for tool in tools if tool.name in ALLOWLIST[agent]]
