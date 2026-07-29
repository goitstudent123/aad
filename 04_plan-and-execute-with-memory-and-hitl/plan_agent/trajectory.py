"""Дрібні помічники для артефактів: підпис tool call та запис JSON."""

import json
from pathlib import Path


def signature(tool_call: dict) -> str:
    """Підпис виклику інструменту: ім'я + нормалізовані аргументи."""
    args = json.dumps(tool_call["args"], sort_keys=True, ensure_ascii=False)
    return f"{tool_call['name']}({args})"


def save_json(path: str | Path, data) -> None:
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def merge_json(path: str | Path, key: str, value) -> None:
    """Дописує один розділ у JSON-файл. Демо запускаються окремими процесами, тому
    кожен процес доповнює спільний файл, а не перезаписує його."""
    file = Path(path)
    data = json.loads(file.read_text(encoding="utf-8")) if file.exists() else {}
    data[key] = value
    save_json(file, data)
