"""Чотири інструменти тревел-планера: geocode, weather, currency, facts.

Кожен інструмент = Pydantic v2 схема (Field з description + field_validator) та функція,
яка ходить у публічний API без ключа. Помилки мережі повертаються як {"error": ...},
щоб агент міг спробувати інший шлях замість падіння графа.
"""

from urllib.parse import quote

import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from .config import HTTP_TIMEOUT

# Вікіпедія просить представлятися власним User-Agent.
_HEADERS = {"User-Agent": "aad-hw2-travel-agent/1.0 (coursework)"}

# ponytail: лише найчастіші коди WMO; решта показується як "код N".
_WMO = {
    0: "ясно",
    1: "переважно ясно",
    2: "мінлива хмарність",
    3: "хмарно",
    45: "туман",
    48: "паморозь",
    51: "мряка",
    53: "мряка",
    55: "сильна мряка",
    61: "невеликий дощ",
    63: "дощ",
    65: "сильний дощ",
    71: "невеликий сніг",
    73: "сніг",
    75: "сильний сніг",
    80: "злива",
    81: "злива",
    82: "сильна злива",
    95: "гроза",
}


def _describe(code: int | None) -> str:
    return _WMO.get(code, f"код {code}")


def _get(url: str, params: dict | None = None) -> dict:
    """GET з тайм-аутом; будь-яка мережева біда стає зрозумілою помилкою для агента."""
    try:
        response = requests.get(url, params=params, headers=_HEADERS, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        return {"error": f"Не вдалося звернутися до {url}: {exc}"}
    except ValueError as exc:
        return {"error": f"Некоректна відповідь від {url}: {exc}"}


class GeocodeArgs(BaseModel):
    city: str = Field(description="Назва міста українською або англійською, наприклад 'Краків'")

    @field_validator("city")
    @classmethod
    def check_city(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Назва міста не може бути порожньою")
        if value.isdigit():
            raise ValueError("Назва міста не може складатися лише з цифр")
        if len(value) > 100:
            raise ValueError("Назва міста задовга (максимум 100 символів)")
        return value


@tool("geocode", args_schema=GeocodeArgs)
def geocode(city: str) -> dict:
    """Знаходить координати міста за його назвою.

    Повертає latitude, longitude, країну та офіційну назву. Викликай першим,
    якщо потрібна погода, бо weather приймає лише координати.
    """
    data = _get(
        "https://geocoding-api.open-meteo.com/v1/search",
        {"name": city, "count": 1, "language": "uk", "format": "json"},
    )
    if "error" in data:
        return data
    results = data.get("results") or []
    if not results:
        return {"error": f"Місто '{city}' не знайдено. Перевір написання."}
    top = results[0]
    return {
        "name": top.get("name"),
        "country": top.get("country"),
        "latitude": top.get("latitude"),
        "longitude": top.get("longitude"),
    }


class WeatherArgs(BaseModel):
    latitude: float = Field(ge=-90, le=90, description="Широта у градусах, від -90 до 90")
    longitude: float = Field(ge=-180, le=180, description="Довгота у градусах, від -180 до 180")
    days: int = Field(default=3, description="Скільки днів прогнозу повернути, від 1 до 7")

    @field_validator("days")
    @classmethod
    def check_days(cls, value: int) -> int:
        if not 1 <= value <= 7:
            raise ValueError("Кількість днів прогнозу має бути від 1 до 7")
        return value


@tool("weather", args_schema=WeatherArgs)
def weather(latitude: float, longitude: float, days: int = 3) -> dict:
    """Повертає поточну погоду та прогноз на кілька днів за координатами.

    Координати бери з інструменту geocode. Для питання «куди їхати на вихідні»
    достатньо days=3.
    """
    data = _get(
        "https://api.open-meteo.com/v1/forecast",
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,weather_code",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
            "forecast_days": days,
            "timezone": "auto",
        },
    )
    if "error" in data:
        return data
    current = data.get("current", {})
    daily = data.get("daily", {})
    return {
        "current": {
            "temperature_c": current.get("temperature_2m"),
            "condition": _describe(current.get("weather_code")),
        },
        "daily": [
            {
                "date": date,
                "min_c": daily["temperature_2m_min"][i],
                "max_c": daily["temperature_2m_max"][i],
                "precipitation_mm": daily["precipitation_sum"][i],
                "condition": _describe(daily["weather_code"][i]),
            }
            for i, date in enumerate(daily.get("time", []))
        ],
    }


class CurrencyArgs(BaseModel):
    amount: float = Field(gt=0, description="Сума в іноземній валюті, більша за нуль")
    code: str = Field(description="Код валюти за ISO 4217, наприклад USD, EUR, PLN")

    @field_validator("code")
    @classmethod
    def check_code(cls, value: str) -> str:
        value = value.strip().upper()
        if len(value) != 3 or not value.isalpha():
            raise ValueError("Код валюти має складатися з трьох літер, наприклад USD")
        if value == "UAH":
            raise ValueError("UAH — базова валюта, конвертувати нема чого")
        return value


@tool("currency", args_schema=CurrencyArgs)
def currency(amount: float, code: str) -> dict:
    """Переводить суму в іноземній валюті у гривні за офіційним курсом НБУ.

    Використовуй, коли питають про бюджет поїздки або вартість у гривнях.
    """
    # НБУ хоче саме `&json` без значення, тому URL збирається вручну.
    data = _get(
        f"https://bank.gov.ua/NBUStatService/v1/statdirectory/exchange?valcode={code}&json"
    )
    if isinstance(data, dict) and "error" in data:
        return data
    if not data:
        return {"error": f"НБУ не має курсу для валюти '{code}'"}
    rate = data[0]["rate"]
    return {
        "amount": amount,
        "code": code,
        "rate": rate,
        "uah": round(amount * rate, 2),
        "date": data[0].get("exchangedate"),
    }


class FactsArgs(BaseModel):
    title: str = Field(description="Назва статті у Вікіпедії, зазвичай назва міста")

    @field_validator("title")
    @classmethod
    def check_title(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Назва статті не може бути порожньою")
        if value.startswith("http"):
            raise ValueError("Потрібна назва статті, а не посилання")
        return value


@tool("facts", args_schema=FactsArgs)
def facts(title: str) -> dict:
    """Дає коротку довідку про місто чи об'єкт з української Вікіпедії.

    Використовуй для контексту: чим відоме місто, що там подивитися.
    """
    data = _get(f"https://uk.wikipedia.org/api/rest_v1/page/summary/{quote(title)}")
    if "error" in data:
        return {"error": f"Статті '{title}' немає в українській Вікіпедії"}
    return {
        "title": data.get("title"),
        "extract": data.get("extract"),
        "url": data.get("content_urls", {}).get("desktop", {}).get("page"),
    }


TOOLS = [geocode, weather, currency, facts]
