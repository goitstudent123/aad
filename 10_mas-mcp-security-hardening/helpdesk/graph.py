"""MAS у LangGraph: supervisor + 3 агенти, guardrails на вході й виході, HITL на ризиковому tool."""

import operator
from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from pydantic import BaseModel, Field

from .config import AGENTS, MAX_HANDOFFS, make_llm
from .guardrails import detect_injection, redact_answer
from .logs import short, trace
from .mcp_client import tools_for
from .prompts import ANSWER, SUPERVISOR, WORKERS


class Routing(BaseModel):
    """Рішення супервізора: хто працює далі."""

    next: Literal["kb_agent", "directory_agent", "ops_agent", "FINISH"] = Field(
        description="Наступний агент або FINISH, якщо зібраних даних достатньо"
    )
    reason: str = Field(description="Коротке пояснення передачі українською")


class HelpdeskAnswer(BaseModel):
    """Фінальна відповідь служби підтримки."""

    summary: str = Field(description="Відповідь користувачу українською, 2-5 речень")
    steps: list[str] = Field(default_factory=list, description="Що зробити користувачу")
    actions: list[str] = Field(
        default_factory=list, description="Виконані незворотні дії або позначка, що їх відхилено"
    )
    warnings: list[str] = Field(default_factory=list, description="Ризики та чого бракує")


class MASState(TypedDict):
    messages: Annotated[list, add_messages]
    route: str
    handoffs: Annotated[list[str], operator.add]
    blocked: Optional[str]
    answer: Optional[dict]
    pii_redacted: list[str]


def _render(message) -> str:
    if isinstance(message, HumanMessage):
        return f"Користувач: {message.content}"
    if isinstance(message, ToolMessage):
        return f"Інструмент {message.name}: {short(message.content, 400)}"
    if isinstance(message, AIMessage):
        calls = ", ".join(f"{c['name']}({c['args']})" for c in message.tool_calls)
        name = getattr(message, "name", None) or "агент"
        return f"Виклик {calls}" if calls else f"{name}: {message.content}"
    return str(message.content)


def transcript(state) -> str:
    return "\n".join(_render(m) for m in state["messages"])


def build_mas(llm, tools, checkpointer=None):
    """Збирає граф: input_guard → supervisor ⇄ агенти → respond."""
    supervisor_llm = llm.with_structured_output(Routing, method="function_calling")
    answer_llm = llm.with_structured_output(HelpdeskAnswer, method="function_calling")

    def input_guard(state: MASState) -> dict:
        query = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), ""
        )
        verdict = detect_injection(query)
        if verdict.blocked:
            trace("guard", f"⛔ {verdict.reason}")
            return {"blocked": verdict.reason, "route": "FINISH"}
        return {"blocked": None}

    async def supervisor(state: MASState) -> dict:
        if len(state["handoffs"]) >= MAX_HANDOFFS:
            trace("superv", f"⛔ ліміт {MAX_HANDOFFS} передач")
            return {"route": "FINISH"}

        decision = await supervisor_llm.ainvoke(
            [
                SystemMessage(SUPERVISOR),
                HumanMessage(
                    f"Хід роботи:\n{transcript(state)}\n\n"
                    f"Вже залучені агенти: {state['handoffs'] or 'жодного'}.\n"
                    "Хто працює далі?"
                ),
            ]
        )
        if decision is None:
            return {"route": "FINISH"}

        trace("superv", f"{decision.next} — {short(decision.reason, 120)}")
        return {
            "route": decision.next,
            "handoffs": [decision.next] if decision.next != "FINISH" else [],
        }

    async def respond(state: MASState) -> dict:
        if state.get("blocked"):
            answer = {
                "summary": "Запит не виконано: він містить ознаки атаки на інструкції агента.",
                "steps": ["Сформулюй питання без спроб змінити правила роботи агента"],
                "actions": [],
                "warnings": [state["blocked"]],
            }
            return {"answer": answer, "pii_redacted": []}

        result = await answer_llm.ainvoke(
            [SystemMessage(ANSWER), HumanMessage(f"Хід роботи:\n{transcript(state)}")]
        )
        answer = result.model_dump() if result else {
            "summary": transcript(state),
            "warnings": ["Провайдер не віддав структуровану відповідь"],
        }
        clean, found = redact_answer(answer)
        if found:
            trace("output", f"PII приховано: {', '.join(found)}")
        return {"answer": clean, "pii_redacted": found}

    graph = StateGraph(MASState)
    graph.add_node("input_guard", input_guard)
    graph.add_node("supervisor", supervisor)
    for name in AGENTS:
        graph.add_node(
            name,
            create_react_agent(llm, tools_for(name, tools), prompt=WORKERS[name], name=name),
        )
    graph.add_node("respond", respond)

    graph.add_edge(START, "input_guard")
    graph.add_conditional_edges(
        "input_guard",
        lambda s: "respond" if s.get("blocked") else "supervisor",
        {"respond": "respond", "supervisor": "supervisor"},
    )
    graph.add_conditional_edges(
        "supervisor",
        lambda s: "respond" if s["route"] == "FINISH" else s["route"],
        {**{name: name for name in AGENTS}, "respond": "respond"},
    )
    for name in AGENTS:
        graph.add_edge(name, "supervisor")
    graph.add_edge("respond", END)
    return graph.compile(checkpointer=checkpointer)


def _result(state: dict) -> dict:
    """Стан графа → компактний результат для демонстрацій і артефактів."""
    pending = [i.value for i in state.get("__interrupt__", ())]
    return {
        "answer": state.get("answer"),
        "handoffs": state.get("handoffs", []),
        "pii_redacted": state.get("pii_redacted", []),
        "blocked": state.get("blocked"),
        "pending_approval": pending,
        "tool_calls": [
            f"{c['name']}({c['args']})"
            for m in state.get("messages", [])
            if isinstance(m, AIMessage)
            for c in m.tool_calls
        ],
        "transcript": [_render(m) for m in state.get("messages", [])],
    }


async def run_mas(app, query: str, config: dict) -> dict:
    """Один прогін MAS. Якщо агент дійшов до ризикової дії — повертає pending_approval."""
    state = await app.ainvoke(
        {
            "messages": [HumanMessage(query)],
            "route": "",
            "handoffs": [],
            "blocked": None,
            "answer": None,
            "pii_redacted": [],
        },
        {**config, "recursion_limit": 40},
    )
    return {"query": query, **_result(state)}


async def resume_mas(app, config: dict, approved: bool, comment: str = "") -> dict:
    """Продовження після HITL-паузи: рішення людини повертається у той самий interrupt."""
    state = await app.ainvoke(
        Command(resume={"approved": approved, "comment": comment}),
        {**config, "recursion_limit": 40},
    )
    return {"approved": approved, **_result(state)}


async def build_default(checkpointer=None):
    """Готовий до роботи граф із MCP-інструментами."""
    from .mcp_client import load_mcp_tools

    return build_mas(make_llm(), await load_mcp_tools(), checkpointer=checkpointer)
