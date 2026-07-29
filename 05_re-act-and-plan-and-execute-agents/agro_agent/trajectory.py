"""Траєкторія виконання та запис артефактів."""

import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def signature(tool_call: dict) -> str:
    """Підпис виклику: ім'я + нормалізовані аргументи. За ним ловиться зациклення."""
    args = json.dumps(tool_call["args"], sort_keys=True, ensure_ascii=False)
    return f"{tool_call['name']}({args})"


def trajectory_from(messages) -> list[dict]:
    """Історія повідомлень → плаский лог thought / action / observation."""
    steps: list[dict] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            steps.append({"type": "query", "content": message.content})
        elif isinstance(message, AIMessage):
            if message.content:
                steps.append({"type": "thought", "content": message.content})
            for call in message.tool_calls:
                steps.append(
                    {
                        "type": "action",
                        "tool": call["name"],
                        "args": call["args"],
                        "signature": signature(call),
                    }
                )
        elif isinstance(message, ToolMessage):
            steps.append(
                {
                    "type": "observation",
                    "tool": message.name,
                    "content": message.content,
                }
            )
    return [{"step": index, **step} for index, step in enumerate(steps, start=1)]


def save_json(path: str | Path, data) -> None:
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def merge_json(path: str | Path, key: str, value) -> None:
    """Дописує один розділ у JSON-файл: демо запускаються окремими процесами."""
    file = Path(path)
    data = json.loads(file.read_text(encoding="utf-8")) if file.exists() else {}
    data[key] = value
    save_json(file, data)
