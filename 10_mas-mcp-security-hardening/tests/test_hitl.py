"""HITL і обгортка інструментів: без LLM, на маленькому графі з одним вузлом."""

import pytest
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from typing_extensions import TypedDict

from helpdesk.guardrails import ResetArgs
from helpdesk.mcp_client import guarded

ARGS = {"employee_id": "E1001", "reason": "заявка T-2041, особу підтверджено"}


class State(TypedDict):
    result: str


@pytest.fixture
def executed():
    return []


@pytest.fixture
def app(executed):
    async def reset(employee_id: str, reason: str) -> dict:
        executed.append(employee_id)
        return {"employee_id": employee_id, "temporary_password": "qP3s9-Xk2Lm"}

    tool = guarded(
        StructuredTool.from_function(
            coroutine=reset, name="reset_password",
            description="скидає пароль", args_schema=ResetArgs,
        ),
        agent="ops_agent",
    )

    async def node(state: State) -> dict:
        return {"result": str(await tool.ainvoke(ARGS))}

    graph = StateGraph(State)
    graph.add_node("act", node)
    graph.add_edge(START, "act")
    graph.add_edge("act", END)
    return graph.compile(checkpointer=InMemorySaver())


async def test_risky_tool_pauses_before_execution(app, executed):
    config = {"configurable": {"thread_id": "pause"}}
    state = await app.ainvoke({"result": ""}, config)

    assert state["__interrupt__"], "ризиковий інструмент має зупинити граф"
    assert state["__interrupt__"][0].value["tool"] == "reset_password"
    assert executed == [], "до підтвердження інструмент не виконується"


async def test_approval_executes_the_tool(app, executed):
    config = {"configurable": {"thread_id": "approve"}}
    await app.ainvoke({"result": ""}, config)
    state = await app.ainvoke(Command(resume={"approved": True}), config)

    assert executed == ["E1001"]
    assert "temporary_password" in state["result"]


async def test_rejection_keeps_the_tool_unexecuted(app, executed):
    config = {"configurable": {"thread_id": "reject"}}
    await app.ainvoke({"result": ""}, config)
    state = await app.ainvoke(
        Command(resume={"approved": False, "comment": "керівник не підтвердив"}), config
    )

    assert executed == []
    assert "ВІДМОВЛЕНО" in state["result"]


async def test_wrapper_blocks_tool_outside_agent_allowlist(executed):
    async def reset(employee_id: str, reason: str) -> dict:
        executed.append(employee_id)
        return {}

    tool = guarded(
        StructuredTool.from_function(
            coroutine=reset, name="reset_password",
            description="скидає пароль", args_schema=ResetArgs,
        ),
        agent="kb_agent",
    )

    assert "ВІДХИЛЕНО" in await tool.ainvoke(ARGS)
    assert executed == []
