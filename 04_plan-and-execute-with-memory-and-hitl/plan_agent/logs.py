"""Живий лог ходу роботи агента.

Кожен вузол графа пише, що саме він робить, з таймстемпом від старту процесу: інакше
агент десятками секунд молчить у термінал, і незрозуміло, чи він працює, чи завис на
429 від провайдера. У тестах pytest перехоплює stdout, тож нікому не заважає.
"""

import time

_STARTED = time.monotonic()


def trace(node: str, message: str) -> None:
    print(f"  [{time.monotonic() - _STARTED:6.1f}s] {node:9} │ {message}", flush=True)


def short(value, limit: int = 220) -> str:
    """Обрізає довгі значення, щоб один крок лога лишався в один рядок."""
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"
