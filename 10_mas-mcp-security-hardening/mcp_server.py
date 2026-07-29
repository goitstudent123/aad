"""MCP-сервер служби підтримки «Nexora IT Desk»: 4 інструменти поверх локальних даних.

Запуск як окремий процес: `python mcp_server.py` (транспорт stdio).
Файл навмисно нічого не імпортує з пакета helpdesk — сервер має жити самостійно.
"""

from datetime import date, timedelta
from secrets import token_urlsafe

from fastmcp import FastMCP

mcp = FastMCP("nexora-helpdesk")

# Картки співробітників містять PII — саме на них перевіряється output guardrail.
EMPLOYEES = {
    "E1001": {
        "employee_id": "E1001",
        "name": "Олена Ковальчук",
        "email": "olena.kovalchuk@nexora.ua",
        "phone": "+380 (67) 123-45-67",
        "department": "Фінанси",
        "role": "Головний бухгалтер",
        "manager_id": "E1003",
        "devices": ["NB-4412"],
        "mfa_enabled": True,
        "account_locked": True,
    },
    "E1002": {
        "employee_id": "E1002",
        "name": "Ігор Дідух",
        "email": "ihor.didukh@nexora.ua",
        "phone": "+380 (50) 987-65-43",
        "department": "Логістика",
        "role": "Диспетчер",
        "manager_id": "E1003",
        "devices": ["NB-2210", "PH-0031"],
        "mfa_enabled": False,
        "account_locked": False,
    },
    "E1003": {
        "employee_id": "E1003",
        "name": "Марина Соколова",
        "email": "maryna.sokolova@nexora.ua",
        "phone": "+380 (63) 555-11-02",
        "department": "Фінанси",
        "role": "Керівниця відділу",
        "manager_id": None,
        "devices": ["NB-1180"],
        "mfa_enabled": True,
        "account_locked": False,
    },
    "E1004": {
        "employee_id": "E1004",
        "name": "Тарас Бондаренко",
        "email": "taras.bondarenko@nexora.ua",
        "phone": "+380 (98) 400-77-19",
        "department": "IT",
        "role": "Системний адміністратор",
        "manager_id": None,
        "devices": ["NB-0007"],
        "mfa_enabled": True,
        "account_locked": False,
    },
}

TICKETS = [
    {"ticket_id": "T-2041", "employee_id": "E1001", "status": "open", "category": "access",
     "summary": "Обліковий запис заблоковано після 5 невдалих спроб входу", "opened": "2026-07-27"},
    {"ticket_id": "T-2039", "employee_id": "E1001", "status": "closed", "category": "vpn",
     "summary": "VPN розривався щогодини — замінено профіль", "opened": "2026-07-14"},
    {"ticket_id": "T-2044", "employee_id": "E1002", "status": "open", "category": "hardware",
     "summary": "Ноутбук NB-2210 не бачить док-станцію", "opened": "2026-07-28"},
    {"ticket_id": "T-2045", "employee_id": "E1003", "status": "open", "category": "email",
     "summary": "Не приходять листи із зовнішніх доменів", "opened": "2026-07-28"},
]

KB_ARTICLES = [
    {"article_id": "KB-01", "title": "Розблокування облікового запису після невдалих входів",
     "tags": ["пароль", "блокування", "вхід", "обліковий запис", "access"],
     "body": "Обліковий запис блокується на 30 хвилин після 5 невдалих спроб. Якщо працівник "
             "не пам'ятає пароль — потрібне скидання пароля інженером і перевірка особи."},
    {"article_id": "KB-02", "title": "Скидання пароля: політика",
     "tags": ["пароль", "скидання", "політика", "reset"],
     "body": "Тимчасовий пароль дійсний 30 хвилин, змінюється при першому вході. Скидання "
             "виконує лише інженер після підтвердження керівника — дія незворотна."},
    {"article_id": "KB-03", "title": "VPN не підключається",
     "tags": ["vpn", "мережа", "підключення"],
     "body": "Перевір годинник пристрою, перевстанови профіль, потім перевір MFA. "
             "Якщо MFA вимкнено — VPN не пустить."},
    {"article_id": "KB-04", "title": "Налаштування MFA",
     "tags": ["mfa", "двофакторна", "автентифікація", "безпека"],
     "body": "MFA обов'язкова для відділів Фінанси та IT. Реєстрація застосунку — через "
             "портал identity.nexora.ua, резервні коди видає інженер."},
    {"article_id": "KB-05", "title": "Док-станція не бачить ноутбук",
     "tags": ["ноутбук", "док-станція", "hardware", "монітор"],
     "body": "Онови прошивку док-станції, перевір кабель USB-C, від'єднай живлення на 30 секунд."},
    {"article_id": "KB-06", "title": "Не приходить пошта із зовнішніх доменів",
     "tags": ["пошта", "email", "спам", "фільтр"],
     "body": "Перевір карантин у mail.nexora.ua, потім правила транспорту. Зовнішні домени "
             "можуть блокуватися політикою DMARC."},
]


def search_kb(query: str, limit: int = 3) -> list[dict]:
    """Шукає статті бази знань за ключовими словами. Повертає до `limit` статей."""
    words = {w.strip(".,!?:;»«\"'").lower() for w in query.split() if len(w) > 2}

    def score(article: dict) -> int:
        haystack = f"{article['title']} {' '.join(article['tags'])} {article['body']}".lower()
        return sum(1 for word in words if word in haystack)

    ranked = sorted(KB_ARTICLES, key=score, reverse=True)
    return [dict(a, score=score(a)) for a in ranked[:limit] if score(a) > 0]


def lookup_employee(employee_id: str) -> dict:
    """Картка співробітника за табельним номером (формат E1001): контакти, пристрої, стан МФА."""
    employee = EMPLOYEES.get(employee_id.strip().upper())
    if employee is None:
        return {"error": f"Співробітника {employee_id} не знайдено"}
    return dict(employee)


def list_tickets(employee_id: str, status: str = "open") -> list[dict]:
    """Заявки співробітника. status: open, closed або all."""
    employee_id = employee_id.strip().upper()
    return [
        dict(t) for t in TICKETS
        if t["employee_id"] == employee_id and status in ("all", t["status"])
    ]


def reset_password(employee_id: str, reason: str) -> dict:
    """РИЗИКОВИЙ: скидає пароль співробітника й видає тимчасовий. Незворотна дія."""
    employee_id = employee_id.strip().upper()
    if employee_id not in EMPLOYEES:
        return {"error": f"Співробітника {employee_id} не знайдено"}
    EMPLOYEES[employee_id]["account_locked"] = False
    return {
        "employee_id": employee_id,
        "temporary_password": token_urlsafe(9),
        "expires_at": str(date.today() + timedelta(days=1)),
        "must_change_on_login": True,
        "reason": reason,
    }


TOOLS = (search_kb, lookup_employee, list_tickets, reset_password)

for _tool in TOOLS:
    mcp.tool(_tool)

if __name__ == "__main__":
    # Банер FastMCP забруднює stdio-канал у логах агента.
    mcp.run(show_banner=False)
