"""Доменні інструменти агронома. Кожен повертає JSON-рядок {status, data|error}.

locate_field і input_cost мають альтернативні джерела: якщо основний API впав,
call_tool повторює запит через резервне (fallback-стратегія).
"""

import json
from datetime import date

import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field, field_validator

from .config import HTTP_TIMEOUT
from .knowledge import search_knowledge
from .logs import short, trace

_HEADERS = {"User-Agent": "aad-hw5-agro-agent/1.0 (coursework)"}

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
OSM_URL = "https://nominatim.openstreetmap.org/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
NBU_URL = "https://bank.gov.ua/NBUStatService/v1/statdirectory/exchange"
ERAPI_URL = "https://open.er-api.com/v6/latest"


def ok(**data) -> str:
    return json.dumps({"status": "ok", "data": data}, ensure_ascii=False)


def fail(message: str) -> str:
    return json.dumps({"status": "error", "error": message}, ensure_ascii=False)


def is_error(output: str) -> bool:
    try:
        return json.loads(output).get("status") == "error"
    except (json.JSONDecodeError, AttributeError):
        return True


def _get(url: str, params: dict | None = None):
    """GET із тайм-аутом. Повертає (payload, error) — мережева біда стає текстом для агента."""
    try:
        response = requests.get(url, params=params, headers=_HEADERS, timeout=HTTP_TIMEOUT)
        response.raise_for_status()
        return response.json(), None
    except requests.RequestException as exc:
        return None, f"джерело {url} недоступне: {exc}"
    except ValueError as exc:
        return None, f"джерело {url} віддало некоректний JSON: {exc}"


class FieldArgs(BaseModel):
    settlement: str = Field(
        description="Населений пункт, біля якого розташоване поле, наприклад 'Тернопіль'"
    )

    @field_validator("settlement")
    @classmethod
    def check_settlement(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Назва населеного пункту не може бути порожньою")
        if value.isdigit():
            raise ValueError("Назва населеного пункту не може складатися лише з цифр")
        if len(value) > 100:
            raise ValueError("Назва населеного пункту задовга (максимум 100 символів)")
        return value


@tool("locate_field", args_schema=FieldArgs)
def locate_field(settlement: str) -> str:
    """Знаходить координати поля за назвою найближчого населеного пункту.

    Повертає latitude, longitude, район і країну. Викликай першим, бо soil_forecast
    приймає лише координати.
    """
    data, error = _get(
        GEOCODE_URL, {"name": settlement, "count": 1, "language": "uk", "format": "json"}
    )
    if error:
        return fail(error)
    results = data.get("results") or []
    if not results:
        return fail(f"Населений пункт '{settlement}' не знайдено. Перевір написання.")
    top = results[0]
    return ok(
        name=top.get("name"),
        region=top.get("admin1"),
        country=top.get("country"),
        latitude=top.get("latitude"),
        longitude=top.get("longitude"),
        source="open-meteo",
    )


def _locate_via_osm(settlement: str) -> str:
    """Резервний геокодер (OpenStreetMap Nominatim)."""
    data, error = _get(OSM_URL, {"q": settlement, "format": "json", "limit": 1})
    if error:
        return fail(error)
    if not data:
        return fail(f"Резервне джерело також не знайшло '{settlement}'")
    top = data[0]
    return ok(
        name=top.get("name"),
        region=top.get("display_name"),
        country=None,
        latitude=float(top["lat"]),
        longitude=float(top["lon"]),
        source="nominatim",
    )


class SoilArgs(BaseModel):
    latitude: float = Field(ge=-90, le=90, description="Широта поля у градусах, від -90 до 90")
    longitude: float = Field(ge=-180, le=180, description="Довгота поля у градусах, від -180 до 180")
    days: int = Field(default=3, description="Скільки днів прогнозу повернути, від 1 до 7")

    @field_validator("days")
    @classmethod
    def check_days(cls, value: int) -> int:
        if not 1 <= value <= 7:
            raise ValueError("Кількість днів прогнозу має бути від 1 до 7")
        return value


@tool("soil_forecast", args_schema=SoilArgs)
def soil_forecast(latitude: float, longitude: float, days: int = 3) -> str:
    """Живі дані поля: температура і вологість ґрунту, опади, вітер, випаровування.

    Саме за цими числами вирішується, чи можна сіяти, обприскувати або поливати.
    Координати бери з locate_field. У базі знань цих даних немає.
    """
    data, error = _get(
        FORECAST_URL,
        {
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,wind_speed_10m,precipitation",
            "hourly": "soil_temperature_6cm,soil_moisture_3_to_9cm",
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,"
                     "wind_speed_10m_max,et0_fao_evapotranspiration",
            "forecast_days": days,
            "timezone": "auto",
        },
    )
    if error:
        return fail(error)
    current = data.get("current", {})
    hourly = data.get("hourly", {})
    daily = data.get("daily", {})
    return ok(
        current={
            "air_temperature_c": current.get("temperature_2m"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "precipitation_mm": current.get("precipitation"),
            # Ґрунтові показники в open-meteo є лише погодинні, тому беремо перший строк.
            "soil_temperature_6cm_c": (hourly.get("soil_temperature_6cm") or [None])[0],
            "soil_moisture_3_9cm_m3m3": (hourly.get("soil_moisture_3_to_9cm") or [None])[0],
        },
        daily=[
            {
                "date": day,
                "min_c": daily["temperature_2m_min"][i],
                "max_c": daily["temperature_2m_max"][i],
                "precipitation_mm": daily["precipitation_sum"][i],
                "wind_max_kmh": daily["wind_speed_10m_max"][i],
                "evapotranspiration_mm": daily["et0_fao_evapotranspiration"][i],
            }
            for i, day in enumerate(daily.get("time", []))
        ],
    )


class CostArgs(BaseModel):
    amount: float = Field(gt=0, description="Ціна в іноземній валюті, більша за нуль")
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


@tool("input_cost", args_schema=CostArgs)
def input_cost(amount: float, code: str) -> str:
    """Переводить ціну засобів захисту, добрив або насіння у гривні за курсом НБУ.

    Живий курс на сьогодні. Використовуй для будь-яких розрахунків витрат на гектар.
    """
    # НБУ хоче саме `&json` без значення, тому URL збирається вручну.
    data, error = _get(f"{NBU_URL}?valcode={code}&json")
    if error:
        return fail(error)
    if not data:
        return fail(f"НБУ не має курсу для валюти '{code}'")
    rate = data[0]["rate"]
    return ok(
        amount=amount,
        code=code,
        rate=rate,
        uah=round(amount * rate, 2),
        date=data[0].get("exchangedate"),
        source="nbu",
    )


def _rate_via_erapi(amount: float, code: str) -> str:
    """Резервне джерело курсу (exchangerate-api), коли НБУ не відповідає."""
    data, error = _get(f"{ERAPI_URL}/{code.upper()}")
    if error:
        return fail(error)
    rate = (data.get("rates") or {}).get("UAH")
    if not rate:
        return fail(f"Резервне джерело не має курсу для '{code}'")
    return ok(
        amount=amount,
        code=code.upper(),
        rate=rate,
        uah=round(amount * rate, 2),
        date=data.get("time_last_update_utc"),
        source="exchangerate-api",
    )


class SprayingArgs(BaseModel):
    field_name: str = Field(description="Назва або номер поля, наприклад 'Поле 3 біля лісосмуги'")
    product: str = Field(description="Назва препарату, наприклад 'Раундап' або 'Карате Зеон'")
    area_ha: float = Field(gt=0, description="Площа обробки в гектарах, від 0 до 5000")
    date: str = Field(description="Дата обробки у форматі YYYY-MM-DD")
    dose_l_per_ha: float = Field(gt=0, description="Норма витрати препарату, л/га, від 0 до 20")

    @field_validator("field_name", "product")
    @classmethod
    def check_text(cls, value: str) -> str:
        value = value.strip()
        if len(value) < 2:
            raise ValueError("Назва має містити щонайменше два символи")
        return value

    @field_validator("area_ha")
    @classmethod
    def check_area(cls, value: float) -> float:
        if value > 5000:
            raise ValueError("Площа понад 5000 га — схоже на помилку у даних")
        return value

    @field_validator("dose_l_per_ha")
    @classmethod
    def check_dose(cls, value: float) -> float:
        if value > 20:
            raise ValueError("Норма витрати понад 20 л/га перевищує будь-який регламент")
        return value

    @field_validator("date")
    @classmethod
    def check_date(cls, value: str) -> str:
        try:
            date.fromisoformat(value.strip())
        except ValueError:
            raise ValueError("Дата обробки має бути у форматі YYYY-MM-DD") from None
        return value.strip()


@tool("schedule_spraying", args_schema=SprayingArgs)
def schedule_spraying(
    field_name: str, product: str, area_ha: float, date: str, dose_l_per_ha: float
) -> str:
    """Ставить обробку поля пестицидом у наряд бригади. РИЗИКОВА НЕЗВОРОТНА ДІЯ.

    Препарат буде витрачено, поле піде під карантин до кінця періоду очікування,
    а помилкова доза псує врожай. Викликай лише коли агроном прямо просить
    призначити обробку і відомі поле, препарат, площа, дата й норма витрати.
    Перед виконанням граф зупиняється і чекає підтвердження людини.
    """
    # Реального постачальника тут немає — важливий сам шлюз підтвердження.
    order = f"SPR-{date.replace('-', '')}-{abs(hash(field_name)) % 1000:03d}"
    return ok(
        result="scheduled",
        order=order,
        field=field_name,
        product=product,
        area_ha=area_ha,
        date=date,
        dose_l_per_ha=dose_l_per_ha,
        total_product_l=round(area_ha * dose_l_per_ha, 2),
    )


TOOLS = [locate_field, soil_forecast, input_cost, search_knowledge, schedule_spraying]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}

# Інструменти, які не виконуються без підтвердження людини (HITL).
RISKY_TOOLS = {"schedule_spraying"}

# Альтернативні джерела на випадок відмови основного.
FALLBACKS = {"locate_field": _locate_via_osm, "input_cost": _rate_via_erapi}


def call_tool(name: str, args: dict) -> str:
    """Виконує інструмент і, якщо джерело відмовило, пробує альтернативне."""
    tool_fn = TOOLS_BY_NAME.get(name)
    if tool_fn is None:
        return fail(f"Інструмента '{name}' не існує")
    try:
        output = tool_fn.invoke(args)
    except Exception as exc:  # noqa: BLE001 — валідація Pydantic або збій інструменту
        # Невалідні аргументи fallback не врятує, тому одразу віддаємо помилку агенту.
        return fail(f"{type(exc).__name__}: {exc}")
    if is_error(output) and name in FALLBACKS:
        trace("fallback", f"{name}: основне джерело відмовило → пробую резервне")
        try:
            alternative = FALLBACKS[name](**args)
        except Exception as exc:  # noqa: BLE001
            return fail(f"основне і резервне джерела недоступні: {exc}")
        if not is_error(alternative):
            trace("fallback", f"{name}: резервне джерело дало результат {short(alternative, 120)}")
            return alternative
        trace("fallback", f"{name}: резервне джерело теж відмовило")
    return output
