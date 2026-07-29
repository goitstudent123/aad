"""Валідація Pydantic-схем інструментів і structured outputs: коректні та некоректні входи."""

import pytest
from pydantic import ValidationError

from agro_agent.knowledge import SearchArgs
from agro_agent.schemas import Plan, ReplanDecision
from agro_agent.tools import CostArgs, FieldArgs, SoilArgs, SprayingArgs

SPRAYING = {
    "field_name": "Поле 3",
    "product": "Раундап",
    "area_ha": 40.0,
    "date": "2026-08-05",
    "dose_l_per_ha": 2.0,
}


# ── некоректні входи відхиляються ─────────────────────────────────────────────


@pytest.mark.parametrize("settlement", ["", "   ", "12345", "х" * 101])
def test_field_rejects_bad_settlement(settlement):
    with pytest.raises(ValidationError):
        FieldArgs(settlement=settlement)


@pytest.mark.parametrize("days", [0, 8, -1])
def test_soil_rejects_bad_days(days):
    with pytest.raises(ValidationError) as exc:
        SoilArgs(latitude=49.5, longitude=25.6, days=days)
    assert "від 1 до 7" in str(exc.value)


@pytest.mark.parametrize("latitude,longitude", [(91.0, 25.6), (49.5, 181.0)])
def test_soil_rejects_impossible_coordinates(latitude, longitude):
    with pytest.raises(ValidationError):
        SoilArgs(latitude=latitude, longitude=longitude)


@pytest.mark.parametrize("code", ["US", "USDD", "12$", "UAH"])
def test_cost_rejects_bad_currency_code(code):
    with pytest.raises(ValidationError):
        CostArgs(amount=100, code=code)


@pytest.mark.parametrize("amount", [0, -50])
def test_cost_rejects_non_positive_amount(amount):
    with pytest.raises(ValidationError):
        CostArgs(amount=amount, code="USD")


@pytest.mark.parametrize("date", ["05.08.2026", "2026-13-01", "завтра", ""])
def test_spraying_rejects_bad_date(date):
    with pytest.raises(ValidationError) as exc:
        SprayingArgs(**{**SPRAYING, "date": date})
    assert "YYYY-MM-DD" in str(exc.value)


@pytest.mark.parametrize("area", [0, -10, 5001])
def test_spraying_rejects_impossible_area(area):
    with pytest.raises(ValidationError):
        SprayingArgs(**{**SPRAYING, "area_ha": area})


@pytest.mark.parametrize("dose", [0, -1, 25])
def test_spraying_rejects_impossible_dose(dose):
    with pytest.raises(ValidationError):
        SprayingArgs(**{**SPRAYING, "dose_l_per_ha": dose})


def test_spraying_rejects_one_letter_names():
    with pytest.raises(ValidationError):
        SprayingArgs(**{**SPRAYING, "product": "Р"})


@pytest.mark.parametrize("n_results", [0, 6])
def test_search_rejects_bad_n_results(n_results):
    with pytest.raises(ValidationError):
        SearchArgs(query="норми азоту", n_results=n_results)


def test_search_rejects_empty_query():
    with pytest.raises(ValidationError):
        SearchArgs(query="   ")


# ── коректні входи проходять і нормалізуються ─────────────────────────────────


def test_field_trims_whitespace():
    assert FieldArgs(settlement="  Умань ").settlement == "Умань"


def test_cost_upcases_currency_code():
    assert CostArgs(amount=18.5, code=" usd ").code == "USD"


def test_soil_defaults_to_three_days():
    assert SoilArgs(latitude=49.5, longitude=25.6).days == 3


def test_spraying_accepts_valid_order():
    args = SprayingArgs(**SPRAYING)
    assert args.area_ha == 40.0 and args.date == "2026-08-05"


# ── structured outputs ────────────────────────────────────────────────────────


def test_plan_requires_goal_and_steps():
    plan = Plan(goal="перевірити умови", steps=["крок1", "крок2"])
    assert plan.steps == ["крок1", "крок2"]
    with pytest.raises(ValidationError):
        Plan(goal="без кроків")


def test_replan_decision_rejects_unknown_action():
    with pytest.raises(ValidationError):
        ReplanDecision(action="cancel", reasoning="такої дії немає")


def test_replan_decision_updated_steps_are_optional():
    assert ReplanDecision(action="continue", reasoning="план ще актуальний").updated_steps is None
