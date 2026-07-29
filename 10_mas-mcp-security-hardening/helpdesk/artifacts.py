"""Запис артефактів здачі: демонстрації запускаються окремими процесами, тож файл дописуємо."""

import json
from pathlib import Path


def merge_json(path, key: str, value) -> None:
    file = Path(path)
    data = json.loads(file.read_text(encoding="utf-8")) if file.exists() else {}
    data[key] = value
    file.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
