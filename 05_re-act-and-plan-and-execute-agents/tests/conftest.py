"""Спільні фейки: модель без мережі, замокані API, SqliteSaver на тимчасовому файлі."""

import sqlite3

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from agro_agent import tools
from agro_agent.schemas import AgroAnswer, Plan, ReplanDecision

GEOCODE_PAYLOAD = {
    "results": [{"name": "Умань", "admin1": "Черкаська область", "country": "Україна",
                 "latitude": 48.75, "longitude": 30.22}]
}

SPRAYING_ARGS = {
    "field_name": "Поле 3",
    "product": "Раундап",
    "area_ha": 40.0,
    "date": "2026-08-05",
    "dose_l_per_ha": 2.0,
}


class StubStructured:
    """Заміна llm.with_structured_output: віддає підготовлені об'єкти по черзі."""

    def __init__(self, queue, default):
        self.queue, self.default = list(queue), default

    def invoke(self, _messages):
        return self.queue.pop(0) if self.queue else self.default


class StubLLM:
    """Фейкова модель: messages — що вона «вирішує» у кожному виклику."""

    def __init__(self, messages=(), plan=None, decisions=(), answer=None):
        self.messages = list(messages)
        self.plan = plan or Plan(goal="стаб-ціль", steps=["крок1"])
        self.decisions = list(decisions)
        self.answer = answer or AgroAnswer(summary="стаб-відповідь")
        self.calls = 0

    def bind_tools(self, _tools):
        return self

    def with_structured_output(self, schema, **_kwargs):
        if schema is Plan:
            return StubStructured([], self.plan)
        if schema is ReplanDecision:
            return StubStructured(
                self.decisions, ReplanDecision(action="continue", reasoning="стаб")
            )
        return StubStructured([], self.answer)

    def invoke(self, _messages):
        self.calls += 1
        return self.messages.pop(0) if self.messages else AIMessage(content="готово")


def tool_call(name: str, args: dict, call_id: str = "1") -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}]
    )


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Жоден тест не має ходити в мережу."""

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return GEOCODE_PAYLOAD

    monkeypatch.setattr(tools.requests, "get", lambda *a, **kw: FakeResponse())


@pytest.fixture
def saver(tmp_path):
    return SqliteSaver(sqlite3.connect(tmp_path / "state.db", check_same_thread=False))
