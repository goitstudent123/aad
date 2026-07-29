"""Тести ReAct-циклу та його захисних механізмів. LLM і мережа замокані."""

import json

from conftest import SPRAYING_ARGS, StubLLM, tool_call

from agro_agent.react import build_react_graph, run_react


def test_react_loop_calls_tool_and_answers():
    llm = StubLLM(messages=[tool_call("locate_field", {"settlement": "Умань"})])

    result = run_react("де поле біля Умані?", llm=llm)

    assert result["tool_calls"] == ['locate_field({"settlement": "Умань"})']
    assert result["steps"] == 2  # виклик з інструментом + фінальне міркування
    assert result["stop_reason"] is None
    assert result["answer"]["summary"] == "стаб-відповідь"
    types = [step["type"] for step in result["trajectory"]]
    assert types == ["query", "action", "observation", "thought"]


def test_observation_carries_tool_json():
    llm = StubLLM(messages=[tool_call("locate_field", {"settlement": "Умань"})])

    result = run_react("координати", llm=llm)

    observation = json.loads(result["trajectory"][2]["content"])
    assert observation["status"] == "ok"
    assert observation["data"]["latitude"] == 48.75


def test_max_steps_guard_stops_the_loop():
    # Кожен виклик — новий аргумент, щоб спрацював саме ліміт кроків, а не детекція повторів.
    llm = StubLLM(messages=[tool_call("locate_field", {"settlement": f"Село-{i}"}) for i in range(6)])

    result = run_react("нескінченний пошук", max_steps=3, llm=llm)

    assert result["steps"] == 3
    assert result["stop_reason"].startswith("max_steps")
    assert result["answer"] is not None  # відповідь усе одно збирається


def test_loop_detection_stops_repeated_call():
    call = tool_call("locate_field", {"settlement": "Умань"})
    llm = StubLLM(messages=[call, call, call])

    result = run_react("двічі те саме", llm=llm)

    assert result["stop_reason"].startswith("loop_detected")
    assert "locate_field" in result["stop_reason"]
    assert result["steps"] == 2


def test_timeout_guard_stops_before_first_llm_call():
    llm = StubLLM(messages=[tool_call("locate_field", {"settlement": "Умань"})])

    result = run_react("немає часу", timeout=-1.0, llm=llm)

    assert result["steps"] == 0
    assert result["stop_reason"].startswith("timeout")
    assert llm.calls == 0  # до моделі з інструментами справа не дійшла


def test_risky_call_is_not_executed_by_the_loop():
    llm = StubLLM(messages=[tool_call("schedule_spraying", SPRAYING_ARGS)])

    result = run_react("призначай обробку", llm=llm)

    assert [c["name"] for c in result["pending"]] == ["schedule_spraying"]
    # Інструмент не виконувався: спостереження у траєкторії немає.
    assert [s["type"] for s in result["trajectory"]] == ["query", "action"]


def test_graph_is_reusable_between_runs():
    llm = StubLLM()
    graph = build_react_graph(llm)

    first = run_react("перший запит", graph=graph)
    second = run_react("другий запит", graph=graph)

    assert first["query"] != second["query"]
    assert first["steps"] == second["steps"] == 1
