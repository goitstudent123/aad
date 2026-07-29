"""Тести Plan-and-Execute: план, переплановування, HITL через interrupt_before, persistence."""

import sqlite3

from conftest import SPRAYING_ARGS, StubLLM, tool_call
from langgraph.checkpoint.sqlite import SqliteSaver

from agro_agent.plan_execute import build_plan_graph, initial_state
from agro_agent.schemas import Plan, ReplanDecision

SPRAYING_QUERY = "призначай обробку Поля 3"


def _spraying_app(saver, args=None, call_name="schedule_spraying"):
    llm = StubLLM(
        messages=[tool_call(call_name, args or SPRAYING_ARGS)],
        plan=Plan(goal="призначити обробку", steps=["поставити обробку в наряд"]),
    )
    return build_plan_graph(llm, checkpointer=saver), llm


def _config(thread: str) -> dict:
    return {"configurable": {"thread_id": thread}}


# ── planner → executor → replanner ────────────────────────────────────────────


def test_plan_is_executed_step_by_step():
    llm = StubLLM(
        messages=[tool_call("locate_field", {"settlement": "Умань"})],
        plan=Plan(goal="перевірити поле", steps=["знайти координати", "підсумувати"]),
    )
    app = build_plan_graph(llm)

    state = app.invoke(initial_state("що там на полі?"))

    assert state["plan"] == ["знайти координати", "підсумувати"]
    assert state["current_step"] == 2
    assert len(state["results"]) == 2
    assert state["completed"] is True
    assert state["answer"]["summary"] == "стаб-відповідь"
    # Крок виконував вкладений ReAct-агент: його траєкторія попала у спільний лог.
    assert [e["type"] for e in state["log"]][:4] == ["plan", "query", "action", "observation"]
    assert any(e.get("tool") == "locate_field" and e["source"] == "react" for e in state["log"])


def test_replan_replaces_only_remaining_steps():
    llm = StubLLM(
        plan=Plan(goal="ціль", steps=["крок1", "крок2", "крок3"]),
        decisions=[ReplanDecision(action="replan", updated_steps=["новий крок"],
                                  reasoning="решта не потрібна")],
    )
    app = build_plan_graph(llm)

    state = app.invoke(initial_state("запит"))

    assert state["plan"] == ["крок1", "новий крок"]
    assert state["current_step"] == 2
    assert state["completed"] is True


def test_finish_stops_before_remaining_steps():
    llm = StubLLM(
        plan=Plan(goal="ціль", steps=["крок1", "крок2", "крок3"]),
        decisions=[ReplanDecision(action="finish", reasoning="ціль досягнута одразу")],
    )
    app = build_plan_graph(llm)

    state = app.invoke(initial_state("запит"))

    assert state["current_step"] == 1
    assert state["answer"] is not None


def test_max_iterations_guard_stops_endless_replanning():
    llm = StubLLM(plan=Plan(goal="ціль", steps=[f"крок{i}" for i in range(6)]))
    app = build_plan_graph(llm)

    state = app.invoke(initial_state("запит", max_iterations=2))

    assert state["current_step"] == 2
    assert state["stop_reason"].startswith("max_iterations")
    assert state["answer"] is not None


def test_missing_structured_plan_falls_back_to_single_step():
    class BlindLLM(StubLLM):
        def with_structured_output(self, schema, **kwargs):
            if schema is Plan:
                return type("Empty", (), {"invoke": lambda self, _m: None})()
            return super().with_structured_output(schema, **kwargs)

    app = build_plan_graph(BlindLLM())

    state = app.invoke(initial_state("порахуй норму азоту"))

    assert state["plan"] == ["порахуй норму азоту"]
    assert state["completed"] is True


# ── HITL: interrupt_before на ризиковому вузлі ────────────────────────────────


def test_risky_call_interrupts_before_execution(saver):
    app, _ = _spraying_app(saver)
    config = _config("hitl")

    app.invoke(initial_state(SPRAYING_QUERY), config)

    snapshot = app.get_state(config)
    assert snapshot.next == ("risky_act",)
    assert [c["name"] for c in snapshot.values["pending"]] == ["schedule_spraying"]
    assert snapshot.values["results"] == []  # обробку ще не поставлено в наряд


def test_approve_executes_the_tool_without_reselecting_arguments(saver):
    app, llm = _spraying_app(saver)
    config = _config("hitl-approve")
    app.invoke(initial_state(SPRAYING_QUERY), config)
    calls_before = llm.calls

    app.update_state(config, {"approval": {"approved": True}})
    state = app.invoke(None, config)

    # Виконує окремий вузол, тому після resume модель не переобирає аргументи заново.
    assert llm.calls == calls_before
    assert '"result": "scheduled"' in state["results"][0]
    assert "SPR-20260805" in state["results"][0]
    assert state["completed"] is True
    assert [e["type"] for e in state["log"] if e["type"] == "approval"]


def test_reject_does_not_execute_the_tool(saver):
    app, _ = _spraying_app(saver)
    config = _config("hitl-reject")
    app.invoke(initial_state(SPRAYING_QUERY), config)

    app.update_state(config, {"approval": {"approved": False, "reason": "вітер 9 м/с"}})
    state = app.invoke(None, config)

    assert "ВІДХИЛЕНО" in state["results"][0]
    assert "вітер 9 м/с" in state["results"][0]
    assert "scheduled" not in state["results"][0]
    assert [e for e in state["log"] if e["type"] == "rejected"]
    assert state["completed"] is True


def test_resume_without_decision_is_treated_as_refusal(saver):
    app, _ = _spraying_app(saver)
    config = _config("hitl-silent")
    app.invoke(initial_state(SPRAYING_QUERY), config)

    state = app.invoke(None, config)  # людина нічого не відповіла

    assert "ВІДХИЛЕНО" in state["results"][0]
    assert "scheduled" not in state["results"][0]


def test_human_can_edit_arguments_before_approval(saver):
    app, _ = _spraying_app(saver)
    config = _config("hitl-edit")
    app.invoke(initial_state(SPRAYING_QUERY), config)
    edited = {**SPRAYING_ARGS, "area_ha": 20.0, "dose_l_per_ha": 1.5}

    app.update_state(config, {"approval": {"approved": True, "args": edited}})
    state = app.invoke(None, config)

    assert '"total_product_l": 30.0' in state["results"][0]


def test_invalid_arguments_become_a_step_result(saver):
    app, _ = _spraying_app(saver, args={**SPRAYING_ARGS, "dose_l_per_ha": 99.0})
    config = _config("hitl-invalid")
    app.invoke(initial_state(SPRAYING_QUERY), config)

    app.update_state(config, {"approval": {"approved": True}})
    state = app.invoke(None, config)

    assert "ValidationError" in state["results"][0]
    assert state["completed"] is True


def test_safe_tools_do_not_interrupt(saver):
    app, _ = _spraying_app(saver, args={"settlement": "Умань"}, call_name="locate_field")
    config = _config("safe")

    state = app.invoke(initial_state("координати поля"), config)

    assert app.get_state(config).next == ()
    assert state["completed"] is True


# ── Checkpointer: збереження та відновлення стану ─────────────────────────────


def _persistent_app(db_path, llm=None):
    """Новий граф і нове з'єднання до того самого файлу = імітація нового процесу."""
    llm = llm or StubLLM(
        messages=[tool_call("schedule_spraying", SPRAYING_ARGS)],
        plan=Plan(goal="призначити обробку", steps=["поставити обробку в наряд", "підсумувати"]),
    )
    return build_plan_graph(
        llm, checkpointer=SqliteSaver(sqlite3.connect(db_path, check_same_thread=False))
    )


def test_state_is_restored_in_a_new_process(tmp_path):
    db = str(tmp_path / "agent_state.db")
    config = _config("persist")

    first = _persistent_app(db)
    first.invoke(initial_state("призначай обробку"), config)

    # Інший об'єкт графа, інше з'єднання, той самий файл і той самий thread_id.
    second = _persistent_app(db, StubLLM())
    restored = second.get_state(config)
    assert restored.values["plan"] == ["поставити обробку в наряд", "підсумувати"]
    assert restored.next == ("risky_act",)

    second.update_state(config, {"approval": {"approved": True}})
    final = second.invoke(None, config)
    assert final["current_step"] == 2
    assert final["completed"] is True


def test_thread_ids_are_independent(tmp_path):
    db = str(tmp_path / "agent_state.db")
    app = _persistent_app(db, StubLLM(plan=Plan(goal="ціль", steps=["крок1"])))
    first, second = _config("session-001"), _config("session-002")

    app.invoke(initial_state("перший запит"), first)
    assert app.get_state(second).values == {}

    app.invoke(initial_state("другий запит"), second)
    assert app.get_state(first).values["messages"][0].content == "перший запит"
    assert app.get_state(second).values["messages"][0].content == "другий запит"
