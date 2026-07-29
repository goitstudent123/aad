"""HITL і обгортка інструментів: без LLM, на мінімальному графі з InMemorySaver."""

import pytest
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from guardrails import UpdateTicketArgs
from hitl import normalise, resume_command
from mcp_client import guarded

ARGS = {"ticket_id": "TKT-001", "new_status": "closed", "reason": "клієнт підтвердив"}


class State(TypedDict):
    result: str


@pytest.fixture
def executed():
    return []


def make_app(executed, agent="billing"):
    async def update(ticket_id: str, new_status: str, reason: str) -> dict:
        executed.append((ticket_id, new_status))
        return {"updated": ticket_id, "new_status": new_status}

    tool = guarded(
        StructuredTool.from_function(coroutine=update, name="update_ticket_status",
                                     description="змінює статус тікета",
                                     args_schema=UpdateTicketArgs),
        agent=agent,
    )

    async def node(state: State) -> dict:
        return {"result": str(await tool.ainvoke(ARGS))}

    graph = StateGraph(State)
    graph.add_node("act", node)
    graph.add_edge(START, "act")
    graph.add_edge("act", END)
    return graph.compile(checkpointer=InMemorySaver())


async def test_risky_tool_pauses_before_execution(executed):
    app = make_app(executed)
    state = await app.ainvoke({"result": ""}, {"configurable": {"thread_id": "pause"}})

    assert state["__interrupt__"], "ризиковий інструмент має зупинити граф"
    payload = state["__interrupt__"][0].value
    assert payload["tool"] == "update_ticket_status" and payload["agent_name"] == "billing"
    assert executed == [], "до підтвердження інструмент не виконується"


async def test_approve_executes_the_tool(executed):
    app = make_app(executed)
    config = {"configurable": {"thread_id": "approve"}}
    await app.ainvoke({"result": ""}, config)
    state = await app.ainvoke(resume_command("approve"), config)

    assert executed == [("TKT-001", "closed")]
    assert "updated" in state["result"]


async def test_reject_keeps_the_tool_unexecuted(executed):
    app = make_app(executed)
    config = {"configurable": {"thread_id": "reject"}}
    await app.ainvoke({"result": ""}, config)
    state = await app.ainvoke(resume_command("reject", comment="клієнт не підтвердив"), config)

    assert executed == []
    assert "ВІДМОВЛЕНО" in state["result"]


async def test_edit_replaces_arguments_before_execution(executed):
    app = make_app(executed)
    config = {"configurable": {"thread_id": "edit"}}
    await app.ainvoke({"result": ""}, config)
    state = await app.ainvoke(
        resume_command("edit", args={"new_status": "resolved"}), config
    )

    assert executed == [("TKT-001", "resolved")], "виконано виправлені людиною аргументи"
    assert "resolved" in state["result"]


async def test_wrapper_blocks_tool_outside_agent_allowlist(executed):
    app = make_app(executed, agent="researcher")
    state = await app.ainvoke({"result": ""}, {"configurable": {"thread_id": "allowlist"}})

    assert "ВІДХИЛЕНО" in state["result"]
    assert executed == []


def test_decision_formats_are_normalised():
    assert normalise(True, ARGS).approved is True
    assert normalise(False, ARGS).approved is False
    assert normalise({"approved": True}, ARGS).action == "approve"
    assert normalise({"action": "reject"}, ARGS).approved is False
    edited = normalise({"action": "edit", "args": {"new_status": "resolved"}}, ARGS)
    assert edited.approved is True and edited.args["new_status"] == "resolved"
    assert edited.args["ticket_id"] == "TKT-001", "решта аргументів лишається"
