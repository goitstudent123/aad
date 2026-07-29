"""MCP-сервер служби підтримки «Nexora Support»: 5 tools, 2 resources, 2 prompts.

Запуск окремим процесом: `python mcp_server.py` (транспорт stdio).
Нічого не імпортує з решти проєкту — сервер має жити самостійно.
"""

import json
from datetime import datetime, timezone

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="nexora-support",
    instructions="Дані служби підтримки телеком-оператора: тікети, клієнти, нарахування.",
    # Службові логи сервера йдуть у той самий stderr, що й трейс агента.
    log_level="ERROR",
)

# Mock-дані замість CRM/біллінгу. Картки клієнтів містять PII — на них перевіряється
# output guardrail.
CUSTOMERS = {
    "C-100": {
        "customer_id": "C-100",
        "name": "Олег Петренко",
        "tier": "gold",
        "email": "oleh.petrenko@nexora.ua",
        "phone": "+380 (67) 341-22-08",
        "tariff": "Максимальний",
        "monthly_fee": 450,
    },
    "C-101": {
        "customer_id": "C-101",
        "name": "Марія Коваленко",
        "tier": "silver",
        "email": "maria.kovalenko@nexora.ua",
        "phone": "+380 (50) 118-44-92",
        "tariff": "Оптимальний",
        "monthly_fee": 290,
    },
    "C-102": {
        "customer_id": "C-102",
        "name": "Андрій Шевчук",
        "tier": "standard",
        "email": "andrii.shevchuk@nexora.ua",
        "phone": "+380 (63) 907-13-55",
        "tariff": "Базовий",
        "monthly_fee": 180,
    },
}

TICKETS = {
    "TKT-001": {
        "customer_id": "C-100",
        "subject": "Не списано платіж за тариф у вересні",
        "status": "open",
        "priority": "high",
        "category": "billing",
        "opened": "2026-09-28",
    },
    "TKT-002": {
        "customer_id": "C-101",
        "subject": "Пристрій не вмикається після оновлення прошивки, помилка SE-23",
        "status": "in_progress",
        "priority": "medium",
        "category": "tech",
        "opened": "2026-09-27",
    },
    "TKT-003": {
        "customer_id": "C-100",
        "subject": "Роутер втрачає зв'язок щовечора",
        "status": "resolved",
        "priority": "low",
        "category": "tech",
        "opened": "2026-08-11",
    },
    "TKT-004": {
        "customer_id": "C-102",
        "subject": "Повернення коштів за невикористаний період",
        "status": "open",
        "priority": "medium",
        "category": "billing",
        "opened": "2026-09-25",
    },
}

BILLING = {
    ("C-100", "2026-09"): {"charged": 450, "paid": 0, "status": "unpaid", "attempts": 2},
    ("C-100", "2026-08"): {"charged": 450, "paid": 450, "status": "paid", "attempts": 1},
    ("C-101", "2026-09"): {"charged": 290, "paid": 290, "status": "paid", "attempts": 1},
    ("C-102", "2026-09"): {"charged": 180, "paid": 180, "status": "paid", "attempts": 1},
}

VALID_STATUSES = ("open", "in_progress", "resolved", "closed")

FAQ = [
    {"q": "Як скинути пароль до кабінету?",
     "a": "Сторінка входу → «Забули пароль?» → лист із посиланням на email договору."},
    {"q": "Скільки триває повернення коштів?",
     "a": "3–5 банківських днів після підтвердження заявки білінг-агентом."},
    {"q": "Час відповіді підтримки?",
     "a": "gold — до 1 години, silver — до 4 годин, standard — до 24 годин."},
    {"q": "Що означає помилка SE-23?",
     "a": "Прошивка не збіглася з профілем пристрою: потрібен відкат до попередньої версії."},
]

REFUND_POLICY = {
    "title": "Політика повернення коштів",
    "rules": [
        "Кошти за невикористаний період повертаються пропорційно кількості повних днів.",
        "Заявка приймається протягом 30 днів після кінця оплаченого періоду.",
        "Повернення виконується на той самий спосіб оплати, 3–5 банківських днів.",
        "Абонплата за місяць, у якому були звернення до платної техпідтримки, не повертається.",
        "Зміна статусу тікета на closed без повернення коштів вимагає причини в полі reason.",
    ],
}


def _json(payload: dict | list) -> str:
    return json.dumps(payload, ensure_ascii=False)


# ── Tools ──────────────────────────────────────────────────────────────────


@mcp.tool()
def get_ticket(ticket_id: str) -> str:
    """Повертає тікет служби підтримки за ідентифікатором.

    Args:
        ticket_id: Ідентифікатор тікета у форматі TKT-001.

    Returns:
        JSON з полями id, customer_id, subject, status, priority, category, opened
        або {"error": ...}, якщо тікета немає.
    """
    try:
        key = ticket_id.strip().upper()
    except AttributeError:
        return _json({"error": "ticket_id має бути рядком"})
    ticket = TICKETS.get(key)
    if ticket is None:
        return _json({"error": f"Тікет {ticket_id} не знайдено"})
    return _json({"id": key, **ticket})


@mcp.tool()
def search_tickets(customer_id: str, status: str = "all") -> str:
    """Шукає тікети клієнта з фільтром за статусом.

    Args:
        customer_id: Ідентифікатор клієнта у форматі C-100.
        status: open | in_progress | resolved | closed | all (за замовчуванням all).

    Returns:
        JSON {"count": n, "tickets": [...]} або {"error": ...} при невалідному статусі.
    """
    key = str(customer_id).strip().upper()
    status = str(status).strip().lower()
    if status != "all" and status not in VALID_STATUSES:
        return _json({"error": f"Невідомий статус. Допустимі: {['all', *VALID_STATUSES]}"})
    found = [
        {"id": tid, **t}
        for tid, t in TICKETS.items()
        if t["customer_id"] == key and status in ("all", t["status"])
    ]
    return _json({"count": len(found), "tickets": found})


@mcp.tool()
def get_customer(customer_id: str) -> str:
    """Повертає картку клієнта: ім'я, рівень обслуговування, контакти, тариф.

    Увага: відповідь містить персональні дані (email, телефон) — у тексті для клієнта
    вони маскуються output guardrail-ом.

    Args:
        customer_id: Ідентифікатор клієнта у форматі C-100.

    Returns:
        JSON картки клієнта або {"error": ...}, якщо клієнта немає.
    """
    key = str(customer_id).strip().upper()
    customer = CUSTOMERS.get(key)
    if customer is None:
        return _json({"error": f"Клієнта {customer_id} не знайдено"})
    return _json(dict(customer))


@mcp.tool()
def get_billing_summary(customer_id: str, period: str = "2026-09") -> str:
    """Нарахування та платежі клієнта за розрахунковий період.

    Args:
        customer_id: Ідентифікатор клієнта у форматі C-100.
        period: Місяць у форматі YYYY-MM (за замовчуванням 2026-09).

    Returns:
        JSON {customer_id, period, charged, paid, balance, status, attempts}
        або {"error": ...}, якщо даних за період немає.
    """
    key = str(customer_id).strip().upper()
    period = str(period).strip()
    if len(period) != 7 or period[4] != "-" or not period.replace("-", "").isdigit():
        return _json({"error": "period має формат YYYY-MM, наприклад 2026-09"})
    record = BILLING.get((key, period))
    if record is None:
        return _json({"error": f"Немає нарахувань для {customer_id} за {period}"})
    return _json({
        "customer_id": key,
        "period": period,
        "balance": record["paid"] - record["charged"],
        **record,
    })


@mcp.tool()
def update_ticket_status(ticket_id: str, new_status: str, reason: str = "") -> str:
    """РИЗИКОВА ДІЯ: змінює статус тікета. Потребує підтвердження людини (HITL).

    Args:
        ticket_id: Ідентифікатор тікета у форматі TKT-001.
        new_status: open | in_progress | resolved | closed.
        reason: Причина зміни — потрапляє в аудит.

    Returns:
        JSON {updated, old_status, new_status, reason, timestamp} або {"error": ...}.
    """
    key = str(ticket_id).strip().upper()
    new_status = str(new_status).strip().lower()
    if key not in TICKETS:
        return _json({"error": f"Тікет {ticket_id} не знайдено"})
    if new_status not in VALID_STATUSES:
        return _json({"error": f"Невідомий статус. Допустимі: {list(VALID_STATUSES)}"})
    old = TICKETS[key]["status"]
    TICKETS[key]["status"] = new_status
    return _json({
        "updated": key,
        "old_status": old,
        "new_status": new_status,
        "reason": reason,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


# ── Resources ──────────────────────────────────────────────────────────────


@mcp.resource("faq://general")
def faq_resource() -> str:
    """Загальний FAQ служби підтримки — read-only довідник."""
    return _json(FAQ)


@mcp.resource("policy://refund")
def refund_policy_resource() -> str:
    """Політика повернення коштів — read-only довідник для білінг-агента."""
    return _json(REFUND_POLICY)


# ── Prompts ────────────────────────────────────────────────────────────────


@mcp.prompt()
def support_reply(customer_name: str, issue_summary: str, tone: str = "professional") -> str:
    """Шаблон відповіді клієнту.

    Args:
        customer_name: Ім'я клієнта.
        issue_summary: Короткий опис проблеми.
        tone: professional | empathetic | concise.
    """
    tones = {
        "professional": "Сформулюй формальну, чітку відповідь",
        "empathetic": "Сформулюй теплу відповідь із визнанням незручностей",
        "concise": "Сформулюй коротку відповідь без зайвих фраз",
    }
    return (
        f"{tones.get(tone, tones['professional'])} клієнту {customer_name}. "
        f"Тема: {issue_summary}. Назви наступний крок і строк."
    )


@mcp.prompt()
def escalation_note(ticket_id: str, severity: str = "medium") -> str:
    """Шаблон службової записки на ескалацію тікета.

    Args:
        ticket_id: Ідентифікатор тікета.
        severity: low | medium | high.
    """
    return (
        f"Склади записку на ескалацію тікета {ticket_id} (критичність {severity}): "
        "що вже перевірено, чому потрібен інженер другої лінії, який строк реакції."
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
