"""Живий трейс ходу роботи: кожен вузол пише, що робить, із часом від старту процесу."""

import time

_STARTED = time.monotonic()


def trace(node: str, message: str) -> None:
    print(f"  [{time.monotonic() - _STARTED:6.1f}s] {node:9} │ {message}", flush=True)


def short(value, limit: int = 220) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else text[:limit] + "…"
