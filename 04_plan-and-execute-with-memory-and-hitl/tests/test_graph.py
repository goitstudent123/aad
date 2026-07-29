"""Тести графа: план, replanning, HITL та persistence. LLM і мережа підмінені фейками."""

import sqlite3

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from plan_agent import tools
from plan_agent.graph import build_graph, initial_state
from plan_agent.schemas import Plan, ReplanDecision, TravelAnswer

BOOKING_ARGS = {
    "hotel_name": "Rynek Inn",
    "city": "Краків",
    "check_in": "2026-08-10",
    "nights": 2,
    "total_cost": 180.0,
}


class StubStructured:
    """Заміна llm.with_structured_output: віддає підготовлені об'єкти по черзі."""

    def __init__(self, queue, default):
        self.queue, self.default = list(queue), default

    def invoke(self, _messages):
        return self.queue.pop(0) if self.queue else self.default


class StubLLM:
    """Фейкова модель. steps — що вона «вирішує» у кожному виклику executor-а."""

    def __init__(self, plan, steps=(), decisions=()):
        self.plan = plan
        self.steps = list(steps)
        self.decisions = list(decisions)
        self.calls = 0  # скільки разів executor звертався до моделі

    def bind_tools(self, _tools):
        return self

    def with_structured_output(self, schema, **_kwargs):
        if schema is Plan:
            return StubStructured([], self.plan)
        if schema is ReplanDecision:
            return StubStructured(
                self.decisions, ReplanDecision(action="continue", reasoning="стаб")
            )
        return StubStructured([], TravelAnswer(summary="стаб"))

    def invoke(self, _messages):
        self.calls += 1
        return self.steps.pop(0) if self.steps else AIMessage(content="крок закрито міркуванням")


def tool_call(name: str, args: dict, call_id: str = "1") -> AIMessage:
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}]
    )


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """Жоден тест тут не має ходити в мережу."""

    class FakeResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"results": [{"name": "Краків", "country": "Польща", "latitude": 50.06,
                                 "longitude": 19.94}]}

    monkeypatch.setattr(tools.requests, "get", lambda *a, **kw: FakeResponse())


@pytest.fixture
def saver(tmp_path):
    return SqliteSaver(sqlite3.connect(tmp_path / "state.db", check_same_thread=False))


# ── Завдання 1: planner → executor → replanner ────────────────────────────────


def test_plan_is_executed_step_by_step():
    llm = StubLLM(
        Plan(goal="перевірити Краків", steps=["знайти координати", "підсумувати"]),
        steps=[tool_call("geocode", {"city": "Краків"}), AIMessage(content="усе зібрано")],
    )
    app = build_graph(llm)

    state = app.invoke(initial_state("що там у Кракові?"))

    assert state["plan"] == ["знайти координати", "підсумувати"]
    assert state["current_step"] == 2
    assert len(state["results"]) == 2
    assert "geocode" in state["results"][0] and "Польща" in state["results"][0]
    assert state["completed"] is True
    assert state["answer"]["summary"] == "стаб"
    assert [e["type"] for e in state["log"]][:3] == ["plan", "action", "observation"]


def test_replan_replaces_only_remaining_steps():
    llm = StubLLM(
        Plan(goal="ціль", steps=["крок1", "крок2", "крок3"]),
        decisions=[ReplanDecision(action="replan", updated_steps=["новий крок"],
                                  reasoning="решта не потрібна")],
    )
    app = build_graph(llm)

    state = app.invoke(initial_state("запит"))

    # Виконаний крок1 залишився, крок2/крок3 замінені одним новим.
    assert state["plan"] == ["крок1", "новий крок"]
    assert state["current_step"] == 2
    assert state["completed"] is True


def test_finish_stops_before_remaining_steps():
    llm = StubLLM(
        Plan(goal="ціль", steps=["крок1", "крок2", "крок3"]),
        decisions=[ReplanDecision(action="finish", reasoning="ціль досягнута одразу")],
    )
    app = build_graph(llm)

    state = app.invoke(initial_state("запит"))

    assert state["current_step"] == 1  # решта кроків не виконувалась
    assert len(state["plan"]) == 3
    assert state["answer"] is not None


def test_max_iterations_guard_stops_endless_replanning():
    llm = StubLLM(Plan(goal="ціль", steps=[f"крок{i}" for i in range(6)]))
    app = build_graph(llm)

    state = app.invoke(initial_state("запит", max_iterations=2))

    assert state["current_step"] == 2
    assert state["stop_reason"].startswith("max_iterations")
    assert state["answer"] is not None  # відповідь усе одно збирається


def test_missing_structured_output_does_not_break_the_graph():
    """Провайдер відповів текстом замість function call — with_structured_output дав None."""

    class BlindLLM(StubLLM):
        def with_structured_output(self, _schema, **_kwargs):
            return StubStructured([], None)

    app = build_graph(BlindLLM(Plan(goal="ціль", steps=["крок1"])))

    state = app.invoke(initial_state("порахуй бюджет"))

    # План відкатується на сам запит, replanner не вважає роботу завершеною завчасно,
    # а respond віддає сирі результати кроків замість падіння.
    assert state["plan"] == ["порахуй бюджет"]
    assert state["goal"] == "порахуй бюджет"
    assert state["completed"] is True
    assert state["answer"]["warnings"]
    assert state["answer"]["summary"] == state["results"][0]


def test_replanner_without_structured_output_continues_the_plan():
    class OneEyedLLM(StubLLM):
        def with_structured_output(self, schema, **kwargs):
            if schema is ReplanDecision:
                return StubStructured([None], ReplanDecision(action="continue", reasoning="стаб"))
            return super().with_structured_output(schema, **kwargs)

    app = build_graph(OneEyedLLM(Plan(goal="ціль", steps=["крок1", "крок2"])))

    state = app.invoke(initial_state("запит"))

    assert state["current_step"] == 2  # план не обірвався на першому кроці
    assert [e["reasoning"] for e in state["log"] if e["type"] == "decision"][0].startswith(
        "structured output не прийшов"
    )


# ── Завдання 4: HITL ──────────────────────────────────────────────────────────


def _booking_app(saver, args=None, call_name="book_hotel"):
    llm = StubLLM(
        Plan(goal="забронювати готель", steps=["забронювати готель у Кракові"]),
        steps=[tool_call(call_name, args or BOOKING_ARGS)],
    )
    return build_graph(llm, checkpointer=saver), llm


def test_risky_tool_interrupts_the_graph(saver):
    app, _ = _booking_app(saver)
    config = {"configurable": {"thread_id": "hitl"}}

    state = app.invoke(initial_state("забронюй готель"), config)

    payload = state["__interrupt__"][0].value
    assert payload["tool"] == "book_hotel"
    assert payload["args"] == BOOKING_ARGS
    assert "Підтвердіть ризикову дію" in payload["message"]
    # Граф справді стоїть перед виконанням, а не після.
    assert app.get_state(config).next == ("approval",)
    assert state.get("results", []) == []


def test_approve_executes_the_tool_without_calling_llm_again(saver):
    app, llm = _booking_app(saver)
    config = {"configurable": {"thread_id": "hitl-approve"}}
    app.invoke(initial_state("забронюй готель"), config)
    calls_before = llm.calls

    state = app.invoke(Command(resume={"approved": True}), config)

    assert llm.calls == calls_before  # resume не перезапускає крок у LLM
    assert "'status': 'booked'" in state["results"][0]
    assert "WE-КРА-20260810" in state["results"][0]
    assert state["completed"] is True


def test_reject_does_not_execute_the_tool(saver):
    app, _ = _booking_app(saver)
    config = {"configurable": {"thread_id": "hitl-reject"}}
    app.invoke(initial_state("забронюй готель"), config)

    state = app.invoke(Command(resume={"approved": False, "reason": "занадто дорого"}), config)

    assert "ВІДХИЛЕНО" in state["results"][0]
    assert "занадто дорого" in state["results"][0]
    assert "booked" not in state["results"][0]
    assert [e for e in state["log"] if e["type"] == "rejected"]
    assert state["completed"] is True  # агент не падає, а завершує роботу


def test_edit_scenario_uses_human_arguments(saver):
    app, _ = _booking_app(saver)
    config = {"configurable": {"thread_id": "hitl-edit"}}
    app.invoke(initial_state("забронюй готель"), config)
    edited = {**BOOKING_ARGS, "check_in": "2026-08-11", "nights": 1, "total_cost": 95.0}

    state = app.invoke(Command(resume={"approved": True, "args": edited}), config)

    assert "WE-КРА-20260811" in state["results"][0]
    assert "'nights': 1" in state["results"][0]


def test_plain_boolean_resume_also_works(saver):
    app, _ = _booking_app(saver)
    config = {"configurable": {"thread_id": "hitl-bool"}}
    app.invoke(initial_state("забронюй готель"), config)

    state = app.invoke(Command(resume=True), config)

    assert "'status': 'booked'" in state["results"][0]


def test_invalid_tool_arguments_become_an_observation(saver):
    app, _ = _booking_app(saver, args={**BOOKING_ARGS, "nights": 99})
    config = {"configurable": {"thread_id": "hitl-invalid"}}
    app.invoke(initial_state("забронюй готель"), config)

    state = app.invoke(Command(resume={"approved": True}), config)

    # Помилка валідації не вбиває граф — вона стає результатом кроку.
    assert "ValidationError" in state["results"][0]
    assert state["completed"] is True


def test_safe_tools_do_not_ask_for_approval(saver):
    app, _ = _booking_app(saver, args={"city": "Краків"}, call_name="geocode")
    config = {"configurable": {"thread_id": "safe"}}

    state = app.invoke(initial_state("координати Кракова"), config)

    assert "__interrupt__" not in state
    assert "geocode" in state["results"][0]


# ── Завдання 2: persistence ───────────────────────────────────────────────────


def _persistent_app(db_path, llm=None):
    """Новий граф із новим з'єднанням до того самого файлу = імітація нового процесу."""
    llm = llm or StubLLM(
        Plan(goal="ціль", steps=["крок1", "крок2", "крок3"]),
        steps=[tool_call("geocode", {"city": "Краків"})],
    )
    return build_graph(llm, checkpointer=SqliteSaver(sqlite3.connect(db_path, check_same_thread=False)))


def test_state_is_restored_in_a_new_process(tmp_path):
    db = str(tmp_path / "agent_state.db")
    config = {"configurable": {"thread_id": "persist"}}

    first = _persistent_app(db)
    for event in first.stream(initial_state("запит"), config, stream_mode="updates"):
        if "act" in event:
            break  # «падіння процесу» після першого виконаного кроку

    # Інший об'єкт графа, інше з'єднання, той самий файл і той самий thread_id.
    second = _persistent_app(db)
    restored = second.get_state(config)
    assert restored.values["current_step"] == 1
    assert restored.values["plan"] == ["крок1", "крок2", "крок3"]
    assert restored.next == ("replanner",)

    final = second.invoke(None, config)  # None = продовжити з checkpoint-а
    assert final["current_step"] == 3
    assert final["completed"] is True
    assert len(final["results"]) == 3


def test_thread_ids_are_independent(tmp_path):
    db = str(tmp_path / "agent_state.db")
    app = _persistent_app(db)
    first = {"configurable": {"thread_id": "session-001"}}
    second = {"configurable": {"thread_id": "session-002"}}

    app.invoke(initial_state("перший запит"), first)
    assert app.get_state(second).values == {}  # чужа сесія нічого не бачить

    app.invoke(initial_state("другий запит"), second)
    assert app.get_state(first).values["messages"][0].content == "перший запит"
    assert app.get_state(second).values["messages"][0].content == "другий запит"
