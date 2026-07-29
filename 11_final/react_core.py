"""ReAct-цикл з ДЗ1: LLM → tools → LLM, із max_steps, timeout і детектором повторів.

Спільне тіло для tech-агента, researcher-агента (RAG) і кроків Plan-and-Execute.
"""

import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from config import MAX_STEPS, TIMEOUT_SECONDS
from logs import short, trace
from trajectory_logger import signature


async def react_loop(
    llm,
    tools: list,
    system_prompt: str,
    query: str,
    agent: str,
    max_steps: int = MAX_STEPS,
    timeout: float = TIMEOUT_SECONDS,
) -> dict:
    """Один прогін ReAct-агента. Повертає текст, повідомлення циклу й причину зупинки."""
    bound = llm.bind_tools(tools) if tools else llm
    by_name = {tool.name: tool for tool in tools}
    messages = [HumanMessage(query)]
    seen: set[str] = set()
    steps, stop_reason = 0, None
    deadline = time.monotonic() + timeout

    while True:
        if steps >= max_steps:
            stop_reason = f"max_steps: ліміт {max_steps} кроків"
            trace(agent, f"⛔ {stop_reason}")
            break
        if time.monotonic() >= deadline:
            stop_reason = "timeout: вичерпано час агента"
            trace(agent, f"⛔ {stop_reason}")
            break

        reply = await bound.ainvoke([SystemMessage(system_prompt), *messages])
        messages.append(reply)
        steps += 1

        calls = reply.tool_calls
        if not calls:
            trace(agent, f"крок {steps}: відповідь без інструментів")
            break

        signatures = [signature(call) for call in calls]
        repeated = [s for s in signatures if s in seen]
        for sig in signatures:
            trace(agent, f"крок {steps}: {short(sig, 150)}")

        if repeated:
            stop_reason = f"loop_detected: повторний виклик {repeated[0]}"
            trace(agent, f"⛔ {stop_reason}")
            for call in calls:
                messages.append(ToolMessage(content="Зупинено: повторний виклик з тими самими "
                                                    "аргументами.",
                                            name=call["name"], tool_call_id=call["id"]))
            break
        seen.update(signatures)

        for call in calls:
            tool = by_name.get(call["name"])
            if tool is None:
                output = (f"ВІДХИЛЕНО guardrail-ом — інструмент {call['name']} "
                          f"недоступний агенту {agent}")
                trace("guard", f"⛔ {output}")
            else:
                output = await tool.ainvoke(call["args"])
            messages.append(ToolMessage(content=str(output), name=call["name"],
                                        tool_call_id=call["id"]))

    text = next(
        (m.content for m in reversed(messages) if isinstance(m, AIMessage) and m.content),
        "",
    )
    return {
        "text": text or "(агент не сформулював відповідь)",
        "messages": messages,
        "steps": steps,
        "stop_reason": stop_reason,
        "tool_calls": [signature(c) for m in messages if isinstance(m, AIMessage)
                       for c in m.tool_calls],
    }
