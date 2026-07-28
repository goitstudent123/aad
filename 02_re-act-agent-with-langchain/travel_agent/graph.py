"""ReAct-граф у LangGraph: agent → tools → agent → ... → respond.

Захисні механізми живуть у вузлі agent: перед кожним викликом LLM перевіряються
max_steps і timeout, а одразу після — чи не повторює модель той самий tool call.
"""

import operator
import time
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

from .config import MAX_STEPS, TIMEOUT_SECONDS, make_llm
from .tools import TOOLS
from .trajectory import signature, trajectory_from

SYSTEM_PROMPT = """
You are "Weekend Escape", a travel planner that helps people pick a destination for a short trip.

## Language
Reply in the language the user writes in. Default to Ukrainian when it's unclear.

## Voice
Friendly, polite, open. Light humor is welcome — warm, never snarky. Be candid about
what you don't know.

## Tools
- geocode(city)      -> coordinates
- weather(lat, lon)  -> forecast
- currency(amount)   -> UAH value at the NBU rate
- facts(topic)       -> Wikipedia summary

## Rules
1. Before acting, decide what information is missing, then call only the tools that fill a real gap.
2. weather takes coordinates only — always geocode first.
3. Never invent numbers. Temperatures, rates and facts come from tools, nowhere else.
4. Never repeat a tool call with identical arguments; reuse the result you already have.
5. If a tool fails or returns nothing, say so plainly and answer with what you do have.
6. If the request is ambiguous (several cities share a name, no dates given), ask one short
   clarifying question instead of guessing.

## Answer format
- Short and practical: 3-6 bullets.
- Lead with the recommendation, then the supporting facts — weather, rough cost, one
  interesting detail.
- Close with a single practical tip (what to pack, when to set off) when it adds something.
"""


class TravelAnswer(BaseModel):
    """Структурована фінальна відповідь агента."""

    city: str = Field(description="Місто, про яке йдеться; '—' якщо питання не про місто")
    summary: str = Field(description="Коротка відповідь користувачу українською")
    facts: list[str] = Field(
        default_factory=list, description="Ключові факти, отримані з інструментів"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Чого дізнатися не вдалося або що варто перевірити"
    )


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    steps: int
    max_steps: int
    deadline: float
    seen_calls: Annotated[list[str], operator.add]
    stop_reason: Optional[str]
    answer: Optional[dict]


def _render(message) -> str:
    """Повідомлення → рядок для фінального промпта (без tool-call протоколу)."""
    if isinstance(message, HumanMessage):
        return f"Питання користувача: {message.content}"
    if isinstance(message, ToolMessage):
        return f"Спостереження ({message.name}): {message.content}"
    if isinstance(message, AIMessage):
        calls = ", ".join(signature(c) for c in message.tool_calls)
        return f"Дія: {calls}" if calls else f"Міркування: {message.content}"
    return str(message.content)


def build_graph(llm=None):
    """Збирає StateGraph. llm можна підмінити фейком — цим користуються тести."""
    llm = llm or make_llm()
    llm_with_tools = llm.bind_tools(TOOLS)
    llm_structured = llm.with_structured_output(TravelAnswer, method="function_calling")

    def agent_node(state: AgentState) -> dict:
        # Захист 1: ліміт ітерацій циклу.
        if state["steps"] >= state["max_steps"]:
            return {"stop_reason": f"max_steps: досягнуто ліміт у {state['max_steps']} кроків"}
        # Захист 2: загальний тайм-аут виконання.
        if time.monotonic() >= state["deadline"]:
            return {"stop_reason": "timeout: вичерпано загальний час виконання"}

        message = llm_with_tools.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])
        update = {"messages": [message], "steps": state["steps"] + 1}

        # Захист 3: детекція зациклення — той самий інструмент з тими самими аргументами.
        signatures = [signature(call) for call in message.tool_calls]
        repeated = [s for s in signatures if s in state["seen_calls"]]
        if repeated:
            update["stop_reason"] = f"loop_detected: повторний виклик {repeated[0]}"
        update["seen_calls"] = signatures
        return update

    def route(state: AgentState) -> str:
        if state.get("stop_reason"):
            return "respond"
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else "respond"

    def respond_node(state: AgentState) -> dict:
        # Фінальна відповідь збирається з текстового переказу траєкторії, а не з сирої
        # історії: так structured output не спотикається про незакриті tool calls,
        # коли агента зупинив захисний механізм.
        transcript = "\n".join(_render(m) for m in state["messages"])
        prompt = f"Хід роботи агента:\n{transcript}\n\nСформуй фінальну відповідь."
        if state.get("stop_reason"):
            prompt += (
                f"\n\nУВАГА: агента примусово зупинено ({state['stop_reason']}). "
                "Дай найкращу відповідь із зібраного та обов'язково познач у warnings, чого бракує."
            )
        answer = llm_structured.invoke([SystemMessage(SYSTEM_PROMPT), HumanMessage(prompt)])
        return {"answer": answer.model_dump()}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("respond", respond_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", "respond": "respond"})
    graph.add_edge("tools", "agent")
    graph.add_edge("respond", END)
    return graph.compile()


def run_agent(
    query: str,
    max_steps: int = MAX_STEPS,
    timeout: float = TIMEOUT_SECONDS,
    llm=None,
    graph=None,
) -> dict:
    """Проганяє один запит і повертає результат разом із траєкторією."""
    graph = graph or build_graph(llm)
    started = time.monotonic()
    state = graph.invoke(
        {
            "messages": [HumanMessage(query)],
            "steps": 0,
            "max_steps": max_steps,
            "deadline": started + timeout,
            "seen_calls": [],
            "stop_reason": None,
            "answer": None,
        },
        # Підстраховка самого LangGraph на випадок, якщо наші лічильники підведуть.
        {"recursion_limit": 2 * max_steps + 5},
    )
    trajectory = trajectory_from(state["messages"])
    return {
        "query": query,
        "answer": state.get("answer"),
        "steps": state["steps"],
        "stop_reason": state.get("stop_reason"),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "tool_calls": [s["signature"] for s in trajectory if s["type"] == "action"],
        "limits": {"max_steps": max_steps, "timeout_seconds": timeout},
        "trajectory": trajectory,
    }
