"""ReAct-агент у LangGraph: agent → tools → agent → … → respond.

Захисні механізми живуть у вузлі agent: перед викликом LLM перевіряються max_steps і
timeout, а одразу після — чи не повторює модель той самий виклик з тими самими
аргументами. Ризиковий інструмент цикл не виконує: він виходить із циклу як pending,
а рішення про нього ухвалює людина у Plan-and-Execute графі.
"""

import operator
import time
from typing import Annotated, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from .config import MAX_STEPS, TIMEOUT_SECONDS, make_llm
from .logs import short, trace
from .prompts import SYSTEM_PROMPT
from .schemas import AgroAnswer
from .tools import RISKY_TOOLS, TOOLS, call_tool
from .trajectory import signature, trajectory_from


class ReactState(TypedDict):
    messages: Annotated[list, add_messages]
    steps: int
    max_steps: int
    deadline: float
    seen_calls: Annotated[list[str], operator.add]
    pending: list[dict]  # ризикові виклики, які цикл не виконує сам
    stop_reason: Optional[str]
    answer: Optional[dict]


def initial_state(query: str, max_steps: int = MAX_STEPS, timeout: float = TIMEOUT_SECONDS) -> dict:
    return {
        "messages": [HumanMessage(query)],
        "steps": 0,
        "max_steps": max_steps,
        "deadline": time.monotonic() + timeout,
        "seen_calls": [],
        "pending": [],
        "stop_reason": None,
        "answer": None,
    }


def _render(message) -> str:
    """Повідомлення → рядок для фінального промпта, без tool-call протоколу."""
    if isinstance(message, HumanMessage):
        return f"Запит: {message.content}"
    if isinstance(message, ToolMessage):
        return f"Спостереження ({message.name}): {message.content}"
    if isinstance(message, AIMessage):
        calls = ", ".join(signature(c) for c in message.tool_calls)
        return f"Дія: {calls}" if calls else f"Міркування: {message.content}"
    return str(message.content)


def build_react_graph(llm=None, structured: bool = True):
    """Збирає ReAct-граф. structured=False — respond віддає звичайний текст (так дешевше
    для вкладених запусків із Plan-and-Execute)."""
    llm = llm or make_llm()
    llm_with_tools = llm.bind_tools(TOOLS)
    answer_llm = llm.with_structured_output(AgroAnswer, method="function_calling")

    def agent_node(state: ReactState) -> dict:
        if state["steps"] >= state["max_steps"]:
            trace("agent", f"⛔ ліміт {state['max_steps']} кроків")
            return {"stop_reason": f"max_steps: досягнуто ліміт у {state['max_steps']} кроків"}
        if time.monotonic() >= state["deadline"]:
            trace("agent", "⛔ вичерпано час виконання")
            return {"stop_reason": "timeout: вичерпано загальний час виконання"}

        message = llm_with_tools.invoke([SystemMessage(SYSTEM_PROMPT)] + state["messages"])
        update = {"messages": [message], "steps": state["steps"] + 1}

        signatures = [signature(call) for call in message.tool_calls]
        for sig in signatures:
            trace("agent", f"крок {state['steps'] + 1}: {short(sig, 160)}")
        if not signatures:
            trace("agent", f"крок {state['steps'] + 1}: відповідь без інструментів")

        repeated = [s for s in signatures if s in state["seen_calls"]]
        if repeated:
            trace("agent", f"⛔ повторний виклик {short(repeated[0], 120)}")
            update["stop_reason"] = f"loop_detected: повторний виклик {repeated[0]}"
        update["seen_calls"] = signatures

        risky = [c for c in message.tool_calls if c["name"] in RISKY_TOOLS]
        if risky:
            trace("agent", f"⚠ ризиковий виклик {risky[0]['name']} — потрібне рішення людини")
            update["pending"] = risky
        return update

    def tools_node(state: ReactState) -> dict:
        messages = []
        for call in state["messages"][-1].tool_calls:
            output = call_tool(call["name"], call["args"])
            trace("tools", f"{call['name']} → {short(output, 160)}")
            messages.append(
                ToolMessage(content=output, name=call["name"], tool_call_id=call["id"])
            )
        return {"messages": messages}

    def respond_node(state: ReactState) -> dict:
        # Фінальна відповідь збирається з текстового переказу траєкторії, а не з сирої
        # історії: так structured output не спотикається про незакриті tool calls,
        # коли агента зупинив захисний механізм.
        transcript = "\n".join(_render(m) for m in state["messages"])
        prompt = f"Хід роботи:\n{transcript}\n\nСформуй відповідь лише з цих даних."
        if state.get("pending"):
            prompt += (
                "\n\nУВАГА: ризикова дія ще не виконана — вона чекає підтвердження людини. "
                "Не пиши, що обробку призначено."
            )
        if state.get("stop_reason"):
            prompt += (
                f"\n\nУВАГА: агента примусово зупинено ({state['stop_reason']}). "
                "Познач у warnings, чого бракує."
            )
        if not structured:
            text = llm.invoke([SystemMessage(SYSTEM_PROMPT), HumanMessage(prompt)]).content
            return {"answer": {"summary": text or "(без результату)"}}

        answer = answer_llm.invoke([SystemMessage(SYSTEM_PROMPT), HumanMessage(prompt)])
        if answer is None:
            # Провайдер відповів текстом замість function call — краще сирий переказ,
            # ніж втрачена робота агента.
            answer = AgroAnswer(
                summary=transcript,
                warnings=["Провайдер не віддав структуровану відповідь — це сирий хід роботи"],
            )
        return {"answer": answer.model_dump()}

    def route(state: ReactState) -> str:
        if state.get("pending") or state.get("stop_reason"):
            return "respond"
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else "respond"

    graph = StateGraph(ReactState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_node("respond", respond_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", "respond": "respond"})
    graph.add_edge("tools", "agent")
    graph.add_edge("respond", END)
    return graph.compile()


def run_react(
    query: str,
    max_steps: int = MAX_STEPS,
    timeout: float = TIMEOUT_SECONDS,
    llm=None,
    graph=None,
    structured: bool = True,
) -> dict:
    """Один прогін ReAct-агента разом із траєкторією та лічильниками."""
    graph = graph or build_react_graph(llm, structured=structured)
    started = time.monotonic()
    state = graph.invoke(
        initial_state(query, max_steps, timeout),
        # Підстраховка самого LangGraph, якщо наші лічильники підведуть.
        {"recursion_limit": 2 * max_steps + 5},
    )
    trajectory = trajectory_from(state["messages"])
    return {
        "query": query,
        "answer": state.get("answer"),
        "steps": state["steps"],
        "stop_reason": state.get("stop_reason"),
        "pending": state.get("pending", []),
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "tool_calls": [s["signature"] for s in trajectory if s["type"] == "action"],
        "limits": {"max_steps": max_steps, "timeout_seconds": timeout},
        "trajectory": trajectory,
    }
