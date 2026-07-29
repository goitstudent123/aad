"""Тести інструментів: конверт {status, data|error}, розбір відповідей API, fallback."""

import json

import pytest

from agro_agent import tools


class FakeResponse:
    def __init__(self, payload, error=None):
        self._payload = payload
        self._error = error

    def raise_for_status(self):
        if self._error:
            raise self._error

    def json(self):
        return self._payload


@pytest.fixture
def fake_get(monkeypatch):
    """Підміняє requests.get за списком відповідей: по одній на кожен виклик."""
    calls = []

    def _install(*responses):
        queue = list(responses)

        def _get(url, params=None, headers=None, timeout=None):
            calls.append(url)
            payload = queue.pop(0) if len(queue) > 1 else queue[0]
            if isinstance(payload, Exception):
                raise payload
            return FakeResponse(payload)

        monkeypatch.setattr(tools.requests, "get", _get)
        return calls

    return _install


GEOCODE_PAYLOAD = {
    "results": [{"name": "Умань", "admin1": "Черкаська область", "country": "Україна",
                 "latitude": 48.75, "longitude": 30.22}]
}

FORECAST_PAYLOAD = {
    "current": {"temperature_2m": 21.4, "wind_speed_10m": 9.0, "precipitation": 0.0},
    "hourly": {"soil_temperature_6cm": [17.2, 17.5], "soil_moisture_3_to_9cm": [0.19, 0.19]},
    "daily": {
        "time": ["2026-08-05"],
        "temperature_2m_min": [14.1],
        "temperature_2m_max": [27.3],
        "precipitation_sum": [0.0],
        "wind_speed_10m_max": [11.2],
        "et0_fao_evapotranspiration": [4.6],
    },
}


def test_locate_field_returns_ok_envelope(fake_get):
    fake_get(GEOCODE_PAYLOAD)

    output = json.loads(tools.locate_field.invoke({"settlement": "Умань"}))

    assert output["status"] == "ok"
    assert output["data"]["latitude"] == 48.75
    assert output["data"]["source"] == "open-meteo"


def test_locate_field_reports_unknown_settlement(fake_get):
    fake_get({"results": []})

    output = json.loads(tools.locate_field.invoke({"settlement": "Ксйцукенг"}))

    assert output["status"] == "error"
    assert "не знайдено" in output["error"]


def test_network_failure_becomes_error_envelope(fake_get):
    fake_get(tools.requests.RequestException("connection refused"))

    output = json.loads(tools.locate_field.invoke({"settlement": "Умань"}))

    assert output["status"] == "error"
    assert "недоступне" in output["error"]


def test_soil_forecast_extracts_soil_and_daily(fake_get):
    fake_get(FORECAST_PAYLOAD)

    output = json.loads(
        tools.soil_forecast.invoke({"latitude": 48.75, "longitude": 30.22, "days": 1})
    )

    assert output["data"]["current"]["soil_temperature_6cm_c"] == 17.2
    assert output["data"]["current"]["soil_moisture_3_9cm_m3m3"] == 0.19
    assert output["data"]["daily"][0]["evapotranspiration_mm"] == 4.6


def test_input_cost_converts_by_nbu_rate(fake_get):
    fake_get([{"rate": 41.5, "exchangedate": "05.08.2026"}])

    output = json.loads(tools.input_cost.invoke({"amount": 18.0, "code": "usd"}))

    assert output["data"]["uah"] == 747.0
    assert output["data"]["source"] == "nbu"


def test_input_cost_reports_unknown_currency(fake_get):
    fake_get([])

    output = json.loads(tools.input_cost.invoke({"amount": 10.0, "code": "XBT"}))

    assert output["status"] == "error"


def test_schedule_spraying_counts_total_product():
    output = json.loads(
        tools.schedule_spraying.invoke(
            {"field_name": "Поле 3", "product": "Раундап", "area_ha": 40.0,
             "date": "2026-08-05", "dose_l_per_ha": 2.0}
        )
    )

    assert output["data"]["result"] == "scheduled"
    assert output["data"]["total_product_l"] == 80.0
    assert output["data"]["order"].startswith("SPR-20260805")


# ── call_tool: диспетчер із fallback-стратегією ───────────────────────────────


def test_call_tool_validation_error_does_not_reach_fallback(monkeypatch):
    called = []
    monkeypatch.setitem(tools.FALLBACKS, "locate_field", lambda **kw: called.append(kw))

    output = json.loads(tools.call_tool("locate_field", {"settlement": ""}))

    assert output["status"] == "error"
    assert "ValidationError" in output["error"]
    assert called == []  # невалідні аргументи резервне джерело не врятує


def test_call_tool_switches_to_fallback_when_primary_fails(fake_get):
    # Перший запит (основне джерело) падає, другий (резервне) віддає координати.
    fake_get(
        tools.requests.RequestException("dead"),
        [{"name": "Умань", "display_name": "Умань, Черкаська область", "lat": "48.75",
          "lon": "30.22"}],
    )

    output = json.loads(tools.call_tool("locate_field", {"settlement": "Умань"}))

    assert output["status"] == "ok"
    assert output["data"]["source"] == "nominatim"


def test_call_tool_reports_when_both_sources_fail(fake_get):
    fake_get(tools.requests.RequestException("dead"))

    output = json.loads(tools.call_tool("input_cost", {"amount": 18.0, "code": "USD"}))

    assert output["status"] == "error"


def test_call_tool_rejects_unknown_tool():
    output = json.loads(tools.call_tool("plough_everything", {}))

    assert output["status"] == "error"
    assert "не існує" in output["error"]


def test_risky_tool_is_marked():
    assert tools.RISKY_TOOLS == {"schedule_spraying"}
    assert set(tools.TOOLS_BY_NAME) == {
        "locate_field", "soil_forecast", "input_cost", "search_knowledge", "schedule_spraying"
    }
