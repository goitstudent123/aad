"""Тести графа: захисні механізми та траєкторія. LLM і мережа підмінені фейками."""

import itertools

import pytest
from langchain_core.messages import AIMessage

from travel_agent import tools
from travel_agent.graph import TravelAnswer, build_graph, run_agent


class StubStructured:
    """Заміна llm.with_structured_output — завжди повертає готову відповідь."""

    def invoke(self, messages):
        return TravelAnswer(city="—", summary="стаб", facts=[], warnings=[])


class StubLLM:
    """Фейкова модель: віддає заздалегідь підготовлені AIMessage по черзі."""

    def __init__(self, messages):
        self.queue = list(messages)
        self.calls = 0

    def bind_tools(self, _tools):
        return self

    def with_structured_output(self, _schema, **_kwargs):
        return StubStructured()

    def invoke(self, _messages):
        self.calls += 1
        return self.queue.pop(0) if self.queue else AIMessage(content="усе зрозуміло")


def tool_call_message(city: str, call_id: str) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": "geocode", "args": {"city": city}, "id": call_id, "type": "tool_call"}],
    )


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Жоден тест тут не має ходити в мережу."""

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"results": [{"name": "X", "country": "Y", "latitude": 1.0, "longitude": 2.0}]}

    monkeypatch.setattr(tools.requests, "get", lambda *a, **kw: FakeResponse())


def test_max_steps_stops_the_loop():
    # Модель нескінченно просить нові geocode — має спрацювати ліміт кроків.
    cities = (f"Місто-{i}" for i in itertools.count())
    llm = StubLLM([tool_call_message(next(cities), str(i)) for i in range(10)])

    result = run_agent("будь-що", max_steps=3, llm=llm)

    assert result["steps"] == 3
    assert result["stop_reason"].startswith("max_steps")
    assert result["answer"]["summary"] == "стаб"


def test_timeout_stops_before_first_llm_call():
    llm = StubLLM([tool_call_message("Львів", "1")])

    result = run_agent("будь-що", timeout=0.0, llm=llm)

    assert result["steps"] == 0
    assert result["stop_reason"].startswith("timeout")
    assert llm.calls == 0  # до моделі справа не дійшла


def test_repeated_tool_call_is_detected():
    llm = StubLLM([tool_call_message("Львів", "1"), tool_call_message("Львів", "2")])

    result = run_agent("будь-що", max_steps=5, llm=llm)

    assert result["steps"] == 2
    assert "loop_detected" in result["stop_reason"]
    assert "geocode" in result["stop_reason"]


def test_happy_path_records_trajectory():
    llm = StubLLM([tool_call_message("Львів", "1"), AIMessage(content="готово")])

    result = run_agent("Яка погода у Львові?", llm=llm)

    assert result["stop_reason"] is None
    assert result["tool_calls"] == ['geocode({"city": "Львів"})']
    types = [step["type"] for step in result["trajectory"]]
    assert types == ["query", "action", "observation", "thought"]
    assert result["trajectory"][0]["step"] == 1


def test_graph_can_be_reused_between_runs():
    graph = build_graph(StubLLM([]))
    first = run_agent("раз", graph=graph)
    second = run_agent("два", graph=graph)
    # Стан не тече між прогонами: лічильники стартують з нуля щоразу.
    assert first["steps"] == second["steps"] == 1
