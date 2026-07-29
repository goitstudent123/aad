"""Observability: Langfuse або LangSmith плюс локальний запис spans у trace.json.

Обидва інтегруються через змінні оточення, тому вибір робить .env, а не код.
"""

import json
import os
import time

from langchain_core.callbacks import BaseCallbackHandler

from config import ROOT, TRACE_FILE

LANGFUSE_KEYS = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")


def langfuse_enabled() -> bool:
    return all(os.getenv(key) for key in LANGFUSE_KEYS)


def langsmith_enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING") == "true" and bool(os.getenv("LANGSMITH_API_KEY"))


def backend() -> str:
    if langfuse_enabled():
        return "langfuse"
    if langsmith_enabled():
        return "langsmith"  # трасується автоматично, callback не потрібен
    return "локальний trace.json"


def langfuse_handler():
    """CallbackHandler Langfuse або None, якщо ключів чи інтеграції немає."""
    if not langfuse_enabled():
        return None
    try:
        from langfuse.langchain import CallbackHandler
    except ModuleNotFoundError as error:  # інтеграція Langfuse вимагає пакета langchain
        print(f"  Langfuse вимкнено: {error}")
        return None
    return CallbackHandler()


class SpanRecorder(BaseCallbackHandler):
    """Локальний трейс: вузли, LLM-виклики та інструменти з часом і токенами."""

    def __init__(self, name: str = "run"):
        self.name = name
        self.spans: list[dict] = []
        self._open: dict = {}

    def _start(self, kind, label, run_id, parent_run_id):
        self._open[run_id] = {
            "span": kind,
            "name": label,
            "parent": str(parent_run_id) if parent_run_id else None,
            "started": round(time.monotonic(), 3),
        }

    def _end(self, run_id, **extra):
        span = self._open.pop(run_id, None)
        if span is None:
            return
        span["duration_s"] = round(time.monotonic() - span.pop("started"), 3)
        self.spans.append({**span, **extra})

    def on_chain_start(self, serialized, inputs, *, run_id, parent_run_id=None, **kwargs):
        label = kwargs.get("name") or (serialized or {}).get("name") or "chain"
        self._start("chain", label, run_id, parent_run_id)

    def on_chain_end(self, outputs, *, run_id, **kwargs):
        self._end(run_id)

    def on_chain_error(self, error, *, run_id, **kwargs):
        self._end(run_id, error=str(error))

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kwargs):
        self._start("llm", (serialized or {}).get("name", "llm"), run_id, parent_run_id)

    def on_llm_end(self, response, *, run_id, **kwargs):
        usage = (response.llm_output or {}).get("token_usage") or {}
        if not usage:
            generation = response.generations[0][0] if response.generations else None
            usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
        self._end(run_id, tokens={
            "input": usage.get("prompt_tokens") or usage.get("input_tokens", 0),
            "output": usage.get("completion_tokens") or usage.get("output_tokens", 0),
        })

    def on_llm_error(self, error, *, run_id, **kwargs):
        self._end(run_id, error=str(error))

    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kwargs):
        self._start("tool", (serialized or {}).get("name", "tool"), run_id, parent_run_id)

    def on_tool_end(self, output, *, run_id, **kwargs):
        self._end(run_id, output=str(output)[:200])

    def on_tool_error(self, error, *, run_id, **kwargs):
        self._end(run_id, error=str(error))

    def usage(self) -> dict:
        """Сумарні токени й кількість викликів — основа для порівняння фреймворків."""
        llm = [s for s in self.spans if s["span"] == "llm"]
        return {
            "llm_calls": len(llm),
            "tool_calls": sum(1 for s in self.spans if s["span"] == "tool"),
            "input_tokens": sum(s.get("tokens", {}).get("input", 0) for s in llm),
            "output_tokens": sum(s.get("tokens", {}).get("output", 0) for s in llm),
            "seconds": round(sum(s["duration_s"] for s in self.spans if s["parent"] is None), 2),
        }


def run_config(thread_id: str, recorder: SpanRecorder | None = None) -> dict:
    """Конфіг запуску графа: thread для checkpointer + усі доступні трейсери."""
    callbacks = [c for c in (recorder, langfuse_handler()) if c is not None]
    return {"configurable": {"thread_id": thread_id}, "callbacks": callbacks}


def save_trace(recorders: dict, path=TRACE_FILE) -> str:
    """Фрагмент трейсу у файл — його й здаємо разом із роботою."""
    payload = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"runs": {}}
    payload["backend"] = backend()
    for name, recorder in recorders.items():
        payload["runs"][name] = {"usage": recorder.usage(), "spans": recorder.spans}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def export_langfuse_trace(path=ROOT / "langfuse_trace.json", limit: int = 1):
    """Останні трейси з Langfuse у файл — це і є JSON-доказ того, що дашборд їх бачить."""
    if not langfuse_enabled():
        return None

    import httpx

    host = os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com").rstrip("/")
    auth = (os.environ["LANGFUSE_PUBLIC_KEY"], os.environ["LANGFUSE_SECRET_KEY"])
    listing = httpx.get(f"{host}/api/public/traces", params={"limit": limit},
                        auth=auth, timeout=30.0).json()

    traces = []
    for item in listing.get("data", []):
        detail = httpx.get(f"{host}/api/public/traces/{item['id']}", auth=auth,
                           timeout=30.0).json()
        detail["observations"] = [
            {"name": o.get("name"), "type": o.get("type"), "startTime": o.get("startTime"),
             "endTime": o.get("endTime"), "usage": o.get("usageDetails"),
             "input": str(o.get("input"))[:300], "output": str(o.get("output"))[:300]}
            for o in detail.get("observations", [])
        ]
        traces.append(detail)

    path.write_text(json.dumps(traces, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


if __name__ == "__main__":
    print(f"Observability backend: {backend()}")
    print(f"Langfuse export: {export_langfuse_trace()}")
