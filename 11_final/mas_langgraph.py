"""MAS у LangGraph: supervisor + 4 агенти, guardrails на межах, HITL на ризиковому tool.

    START → guard → supervisor ⇄ {billing | tech | researcher | general} → respond → END

billing — Plan-and-Execute підграф (ДЗ2), tech і researcher — ReAct-цикл з ДЗ1
(researcher ходить у ChromaDB), general — fallback. Стан лежить у SqliteSaver.
"""

import asyncio
import operator
from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from config import AGENTS, MAX_HANDOFFS, make_llm, make_saver
from guardrails import LIMITER, input_guardrail, redact_answer
from logs import short, trace
from mcp_client import all_tools, tools_for
from plan_execute import build_plan_subgraph
from prompts import ANSWER, SUPERVISOR, WORKERS
from react_core import react_loop
from trajectory_logger import log_step, save_trajectory, steps_from_messages


class RouteDecision(BaseModel):
    """Рішення супервізора, до якого агента надіслати запит."""

    action: Literal["billing", "tech", "researcher", "general"] = Field(
        description='Цільовий агент або "general" для нерозпізнаних запитів'
    )
    reasoning: str = Field(description="Коротке пояснення вибору українською")
    finished: bool = Field(
        default=True, description="True, якщо після цього агента даних вистачить для відповіді"
    )


class SupportAnswer(BaseModel):
    """Фінальна відповідь служби підтримки."""

    summary: str = Field(description="Відповідь користувачу українською, 2-5 речень")
    steps: list[str] = Field(default_factory=list, description="Що зробити користувачу")
    actions: list[str] = Field(default_factory=list,
                               description="Виконані незворотні дії або позначка про відмову")
    warnings: list[str] = Field(default_factory=list, description="Ризики та чого бракує")


class MASState(TypedDict):
    messages: Annotated[list, add_messages]
    current_agent: str
    plan: list[str]
    current_step: int
    results: Annotated[list[str], operator.add]
    step_count: int
    plan_iterations: int
    trajectory: Annotated[list[dict], operator.add]
    completed: bool
    pending_approval: bool
    # Без reducer-а: список передач має жити рівно один запит, інакше повторний запуск
    # у тому самому thread_id успадкує чужі передачі й супервізор одразу скаже «досить».
    handoffs: list[str]
    session_id: str
    blocked: Optional[str]
    answer: Optional[dict]
    pii_redacted: list[str]


def initial_state(query: str, session_id: str) -> dict:
    return {
        "messages": [HumanMessage(query)],
        "current_agent": "",
        "plan": [], "current_step": 0, "results": [], "step_count": 0, "plan_iterations": 0,
        "trajectory": [], "completed": False, "pending_approval": False,
        "handoffs": [], "session_id": session_id,
        "blocked": None, "answer": None, "pii_redacted": [],
    }


def _render(message) -> str:
    if isinstance(message, HumanMessage):
        return f"Користувач: {message.content}"
    if isinstance(message, ToolMessage):
        return f"Інструмент {message.name}: {short(message.content, 400)}"
    if isinstance(message, AIMessage):
        calls = ", ".join(f"{c['name']}({c['args']})" for c in message.tool_calls)
        return f"Виклик {calls}" if calls else f"{getattr(message, 'name', None) or 'агент'}: " \
                                               f"{message.content}"
    return str(message.content)


def transcript(state) -> str:
    return "\n".join(_render(m) for m in state["messages"])


def build_mas(llm, tools: list, checkpointer=None):
    """Збирає граф MAS: guard → supervisor ⇄ агенти → respond."""
    supervisor_llm = llm.with_structured_output(RouteDecision, method="function_calling")
    answer_llm = llm.with_structured_output(SupportAnswer, method="function_calling")

    def guard(state: MASState) -> dict:
        """Input guardrail і rate-limit — до першого виклику LLM, щоб не платити за атаку."""
        query = next((m.content for m in reversed(state["messages"])
                      if isinstance(m, HumanMessage)), "")
        ok, reason = LIMITER.check(state.get("session_id") or "default")
        if not ok:
            trace("guard", f"⛔ {reason}")
            return {"blocked": reason, "completed": True, "handoffs": [],
                    "trajectory": [log_step("guard", "rate_limit", query, reason)]}

        safe, cleaned = input_guardrail(query)
        if not safe:
            trace("guard", f"⛔ {cleaned}")
            return {"blocked": cleaned, "completed": True, "handoffs": [],
                    "trajectory": [log_step("guard", "input", query, cleaned)]}
        return {"blocked": None, "handoffs": [],
                "trajectory": [log_step("guard", "input", query, "чисто")]}

    async def supervisor(state: MASState) -> dict:
        if len(state["handoffs"]) >= MAX_HANDOFFS:
            trace("supervisor", f"⛔ ліміт {MAX_HANDOFFS} передач")
            return {"completed": True}

        decision = await supervisor_llm.ainvoke([
            SystemMessage(SUPERVISOR),
            HumanMessage(f"Хід роботи:\n{transcript(state)}\n\n"
                         f"Вже залучені агенти: {state['handoffs'] or 'жодного'}.\n"
                         "Хто працює далі?"),
        ])
        if decision is None:
            return {"completed": True}

        # Перший агент відпрацьовує завжди: finished=true на порожній історії — це
        # відповідь моделі без даних. Повтор того самого агента заборонений жорстко:
        # покладатися тут на промпт означає платити за коло викликів кожного разу,
        # коли модель вирішить «уточнити».
        repeat = decision.action in state["handoffs"]
        if state["handoffs"] and (decision.finished or repeat):
            reason = decision.reasoning + (" [повтор агента заблоковано]" if repeat else "")
            trace("supervisor", f"досить даних — {short(reason, 100)}")
            return {"completed": True,
                    "trajectory": [log_step("supervisor", "route", "finish", reason)]}

        trace("supervisor", f"→ {decision.action}: {short(decision.reasoning, 120)}")
        return {
            "current_agent": decision.action,
            "completed": False,
            "handoffs": [*state["handoffs"], decision.action],
            "step_count": state["step_count"] + 1,
            "trajectory": [log_step("supervisor", "route", state["messages"][-1].content,
                                    f"→ {decision.action}: {decision.reasoning}")],
        }

    def react_node(name: str):
        """Агент-вузол на ReAct-циклі з ДЗ1 (tech, researcher)."""

        async def node(state: MASState) -> dict:
            query = next((m.content for m in reversed(state["messages"])
                          if isinstance(m, HumanMessage)), "")
            run = await react_loop(llm, tools_for(name, tools), WORKERS[name], query, agent=name)
            return {
                "step_count": state["step_count"] + run["steps"],
                "results": [f"{name}: {run['text']}"],
                "messages": [AIMessage(content=run["text"], name=name)],
                "trajectory": steps_from_messages(name, "react", run["messages"]),
            }

        return node

    async def general(state: MASState) -> dict:
        reply = await llm.ainvoke([SystemMessage(WORKERS["general"]),
                                   *state["messages"]])
        return {
            "step_count": state["step_count"] + 1,
            "results": [f"general: {reply.content}"],
            "messages": [AIMessage(content=reply.content, name="general")],
            "trajectory": [log_step("general", "answer", "fallback", reply.content)],
        }

    async def respond(state: MASState) -> dict:
        """Фінальна відповідь + output guardrail (маскування PII)."""
        if state.get("blocked"):
            answer = {
                "summary": "Запит не виконано: спрацював захисний механізм.",
                "steps": ["Сформулюй питання без спроб змінити правила роботи агента"],
                "actions": [], "warnings": [state["blocked"]],
            }
            return {"answer": answer, "pii_redacted": [], "completed": True,
                    "trajectory": [log_step("respond", "blocked", state["blocked"])]}

        result = await answer_llm.ainvoke([
            SystemMessage(ANSWER),
            HumanMessage(f"Хід роботи:\n{transcript(state)}"),
        ])
        answer = result.model_dump() if result else {
            "summary": "\n".join(state["results"]) or transcript(state),
            "steps": [], "actions": [],
            "warnings": ["Провайдер не віддав структуровану відповідь"],
        }
        clean, found = redact_answer(answer)
        if found:
            trace("respond", f"PII приховано: {', '.join(found)}")
        return {"answer": clean, "pii_redacted": found, "completed": True,
                "trajectory": [log_step("respond", "answer", "final", clean["summary"], found)]}

    def route(state: MASState) -> str:
        """Conditional edge: completed → respond (і далі END), інакше — обраний агент."""
        if state.get("completed"):
            return "respond"
        return state.get("current_agent") or "general"

    graph = StateGraph(MASState)
    graph.add_node("guard", guard)
    graph.add_node("supervisor", supervisor)
    graph.add_node("billing", build_plan_subgraph(llm, tools_for("billing", tools)))
    graph.add_node("tech", react_node("tech"))
    graph.add_node("researcher", react_node("researcher"))
    graph.add_node("general", general)
    graph.add_node("respond", respond)

    graph.add_edge(START, "guard")
    graph.add_conditional_edges("guard", lambda s: "respond" if s.get("blocked") else "supervisor",
                                {"respond": "respond", "supervisor": "supervisor"})
    graph.add_conditional_edges("supervisor", route,
                                {**{name: name for name in AGENTS}, "respond": "respond"})
    for name in AGENTS:
        graph.add_edge(name, "supervisor")
    graph.add_edge("respond", END)
    return graph.compile(checkpointer=checkpointer)


async def build_default(checkpointer=None, llm=None):
    """Готовий граф із MCP- та локальними інструментами."""
    return build_mas(llm or make_llm(), await all_tools(), checkpointer=checkpointer)


def result_of(state: dict) -> dict:
    """Стан графа → компактний результат для демонстрацій і артефактів."""
    pending = [i.value for i in state.get("__interrupt__", ())]
    trajectory = state.get("trajectory", [])
    return {
        "answer": state.get("answer"),
        "agents_used": state.get("handoffs", []),
        # Виклики інструментів беремо з траєкторії: усередині агентів свій цикл повідомлень,
        # у батьківські messages потрапляє лише підсумок.
        "tools_called": [tool for entry in trajectory if entry["action"] == "action"
                         for tool in entry["tools"]],
        "step_count": state.get("step_count", 0),
        "pii_redacted": state.get("pii_redacted", []),
        "blocked": state.get("blocked"),
        "pending_approval": pending,
        "trajectory": trajectory,
    }


async def run_mas(app, query: str, config: dict) -> dict:
    """Один прогін MAS. Якщо агент дійшов до ризикової дії — повертає pending_approval."""
    thread = config["configurable"]["thread_id"]
    state = await app.ainvoke(initial_state(query, thread), {**config, "recursion_limit": 40})
    return {"query": query, **result_of(state)}


def print_result(result: dict) -> None:
    answer = result.get("answer") or {}
    print(f"\n  Агенти: {result.get('agents_used')} | інструменти: {result.get('tools_called')}")
    if result.get("pending_approval"):
        print(f"  ⏸ Чекає підтвердження: {result['pending_approval']}")
    print(f"  Відповідь: {short(answer.get('summary'), 600)}")
    for step in answer.get("steps", []):
        print(f"    • {short(step, 200)}")
    if answer.get("actions"):
        print(f"  Дії: {answer['actions']}")
    if answer.get("warnings"):
        print(f"  Застереження: {answer['warnings']}")


DEMO_QUERIES = [
    ("demo-billing", "Не списано платіж за тариф у вересні, тікет TKT-001, клієнт C-100"),
    ("demo-tech", "Пристрій не вмикається після оновлення прошивки, помилка SE-23, тікет TKT-002"),
    ("demo-researcher", "Які правила повернення коштів за невикористаний період?"),
]


async def demo() -> dict:
    """Демонстрація маршрутизації на трьох запитах різного типу."""
    results = {}
    async with make_saver() as saver:
        app = await build_default(checkpointer=saver)
        for thread, query in DEMO_QUERIES:
            print(f"\n── {thread}: {query}")
            result = await run_mas(app, query, {"configurable": {"thread_id": thread}})
            print_result(result)
            save_trajectory(result["trajectory"], key=thread)
            results[thread] = result
    return results


if __name__ == "__main__":
    asyncio.run(demo())
