"""TrajectoryLogger з ДЗ1, розширений полем agent_name: у MAS видно, хто саме зробив крок."""

import json
import time
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from config import TRAJECTORY_FILE


def signature(tool_call: dict) -> str:
    """Підпис виклику: ім'я + нормалізовані аргументи. За ним ловиться зациклення."""
    args = json.dumps(tool_call["args"], sort_keys=True, ensure_ascii=False)
    return f"{tool_call['name']}({args})"


def log_step(agent: str, node: str, action: str, output: str = "", tools: list | None = None) -> dict:
    """Один запис траєкторії MAS."""
    return {
        "agent_name": agent,
        "node": node,
        "action": str(action)[:200],
        "output": str(output)[:300],
        "tools": tools or [],
        "timestamp": round(time.time(), 3),
    }


def steps_from_messages(agent: str, node: str, messages) -> list[dict]:
    """Повідомлення ReAct-циклу → записи траєкторії з agent_name."""
    steps = []
    for message in messages:
        if isinstance(message, HumanMessage):
            steps.append(log_step(agent, node, "query", message.content))
        elif isinstance(message, AIMessage):
            if message.tool_calls:
                steps.append(log_step(
                    agent, node, "action",
                    "; ".join(signature(c) for c in message.tool_calls),
                    [c["name"] for c in message.tool_calls],
                ))
            elif message.content:
                steps.append(log_step(agent, node, "thought", message.content))
        elif isinstance(message, ToolMessage):
            steps.append(log_step(agent, node, "observation", message.content, [message.name]))
    return steps


def dedupe(trajectory: list[dict]) -> list[dict]:
    """Після resume LangGraph переграє вузли, і однакові записи потрапляють у лог двічі."""
    seen, unique = set(), []
    for entry in trajectory:
        key = json.dumps(entry, sort_keys=True, ensure_ascii=False, default=str)
        if key not in seen:
            seen.add(key)
            unique.append(entry)
    return unique


def save_trajectory(trajectory: list[dict], path=TRAJECTORY_FILE, key: str | None = None) -> str:
    """Пише trajectory.json. Прогони дописуються — демо запускаються окремо."""
    file = Path(path)
    payload = json.loads(file.read_text(encoding="utf-8")) if file.exists() else {}
    payload[key or f"run-{len(payload) + 1}"] = dedupe(trajectory)
    file.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                    encoding="utf-8")
    return str(file)
