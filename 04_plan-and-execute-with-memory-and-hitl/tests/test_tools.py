"""Юніт-тести інструментів: валідація Pydantic та розбір відповідей API (мережа замокана)."""

import pytest
from pydantic import ValidationError

from plan_agent import tools
from plan_agent.tools import BookHotelArgs, CurrencyArgs, GeocodeArgs, WeatherArgs


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


@pytest.fixture
def fake_get(monkeypatch):
    """Підміняє requests.get і запам'ятовує, з чим його викликали."""
    calls = []

    def _install(payload):
        def _get(url, params=None, headers=None, timeout=None):
            calls.append((url, params))
            return FakeResponse(payload)

        monkeypatch.setattr(tools.requests, "get", _get)
        return calls

    return _install


# --- валідація: некоректні входи відхиляються зі зрозумілим повідомленням ---


@pytest.mark.parametrize("city", ["", "   ", "12345"])
def test_geocode_rejects_bad_city(city):
    with pytest.raises(ValidationError):
        GeocodeArgs(city=city)


def test_geocode_trims_whitespace():
    assert GeocodeArgs(city="  Львів ").city == "Львів"


@pytest.mark.parametrize("days", [0, 8, -1])
def test_weather_rejects_bad_days(days):
    with pytest.raises(ValidationError) as exc:
        WeatherArgs(latitude=50.0, longitude=30.0, days=days)
    assert "від 1 до 7" in str(exc.value)


def test_weather_rejects_impossible_latitude():
    with pytest.raises(ValidationError):
        WeatherArgs(latitude=91.0, longitude=30.0)


@pytest.mark.parametrize("code", ["US", "USDD", "12A", "UAH"])
def test_currency_rejects_bad_code(code):
    with pytest.raises(ValidationError):
        CurrencyArgs(amount=10, code=code)


def test_currency_normalizes_code():
    assert CurrencyArgs(amount=10, code=" usd ").code == "USD"


def test_currency_rejects_non_positive_amount():
    with pytest.raises(ValidationError):
        CurrencyArgs(amount=0, code="USD")


@pytest.mark.parametrize("check_in", ["10.08.2026", "2026-13-01", "завтра"])
def test_book_hotel_rejects_bad_date(check_in):
    with pytest.raises(ValidationError) as exc:
        BookHotelArgs(hotel_name="Rynek Inn", city="Краків", check_in=check_in, nights=2,
                      total_cost=180)
    assert "YYYY-MM-DD" in str(exc.value)


@pytest.mark.parametrize("nights", [0, 31])
def test_book_hotel_rejects_bad_nights(nights):
    with pytest.raises(ValidationError):
        BookHotelArgs(hotel_name="Rynek Inn", city="Краків", check_in="2026-08-10",
                      nights=nights, total_cost=180)


def test_book_hotel_rejects_free_booking():
    with pytest.raises(ValidationError):
        BookHotelArgs(hotel_name="Rynek Inn", city="Краків", check_in="2026-08-10", nights=2,
                      total_cost=0)


# --- виклик інструменту через LangChain: та сама валідація працює і там ---


def test_tool_invoke_validates_args():
    with pytest.raises(ValidationError):
        tools.geocode.invoke({"city": ""})


def test_risky_tool_is_registered_as_risky():
    assert tools.RISKY_TOOLS == {"book_hotel"}
    assert set(tools.RISKY_TOOLS) <= set(tools.TOOLS_BY_NAME)


# --- розбір відповідей API ---


def test_geocode_parses_first_result(fake_get):
    fake_get({"results": [{"name": "Краків", "country": "Польща", "latitude": 50.06,
                           "longitude": 19.94}]})
    assert tools.geocode.invoke({"city": "Краків"}) == {
        "name": "Краків",
        "country": "Польща",
        "latitude": 50.06,
        "longitude": 19.94,
    }


def test_geocode_reports_unknown_city(fake_get):
    fake_get({"results": []})
    assert "не знайдено" in tools.geocode.invoke({"city": "Ксйцукенг"})["error"]


def test_weather_flattens_forecast(fake_get):
    fake_get(
        {
            "current": {"temperature_2m": 21.3, "weather_code": 0},
            "daily": {
                "time": ["2026-07-28"],
                "temperature_2m_max": [27.0],
                "temperature_2m_min": [15.0],
                "precipitation_sum": [0.0],
                "weather_code": [61],
            },
        }
    )
    result = tools.weather.invoke({"latitude": 50.06, "longitude": 19.94, "days": 1})
    assert result["current"] == {"temperature_c": 21.3, "condition": "ясно"}
    assert result["daily"][0]["condition"] == "невеликий дощ"


def test_currency_converts_to_uah(fake_get):
    calls = fake_get([{"rate": 41.5, "cc": "USD", "exchangedate": "28.07.2026"}])
    result = tools.currency.invoke({"amount": 100, "code": "USD"})
    assert result["uah"] == 4150.0
    assert "valcode=USD" in calls[0][0]


def test_currency_reports_unknown_valcode(fake_get):
    fake_get([])
    assert "не має курсу" in tools.currency.invoke({"amount": 5, "code": "XXX"})["error"]


def test_book_hotel_returns_reference():
    result = tools.book_hotel.invoke(
        {"hotel_name": "Rynek Inn", "city": "Краків", "check_in": "2026-08-10", "nights": 2,
         "total_cost": 180.0}
    )
    assert result["status"] == "booked"
    assert result["reference"] == "WE-КРА-20260810"


def test_network_error_becomes_readable_message(monkeypatch):
    def _boom(*args, **kwargs):
        raise tools.requests.RequestException("з'єднання відхилено")

    monkeypatch.setattr(tools.requests, "get", _boom)
    assert "Не вдалося звернутися" in tools.geocode.invoke({"city": "Львів"})["error"]
