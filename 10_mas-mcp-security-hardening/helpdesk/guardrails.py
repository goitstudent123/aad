"""Три рівні захисту: вхід (ін'єкції), інструменти (allowlist + аргументи), вихід (PII)."""

import re
from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from .config import ALLOWLIST


class GuardrailError(RuntimeError):
    """Виклик інструмента відхилено захисним механізмом."""


# --- input guardrail -------------------------------------------------------

INJECTION_RULES = {
    "override_instructions": r"(ignore|disregard|forget)\s+(all\s+|your\s+)?(the\s+)?"
                             r"(previous|prior|above|earlier|system)\s+"
                             r"(instructions?|prompts?|rules?)"
                             r"|(забудь|ігноруй|відкинь)\s+(усі\s+|всі\s+)?(попередні\s+|свої\s+)?"
                             r"(інструкції|правила|вказівки)",
    "prompt_leak": r"(system\s*prompt|reveal\s+your\s+(instructions|rules))"
                   r"|(системн\w+\s+промпт|покажи\s+(свої\s+)?(інструкції|правила|промпт))",
    "persona_switch": r"(you\s+are\s+now|act\s+as\s+if|developer\s+mode|\bDAN\b)"
                      r"|(ти\s+тепер|вдай,?\s+що\s+ти|режим\s+розробника)",
    "secret_exfiltration": r"(api[\s_-]?key|access\s+token|credentials|\.env\b)"
                           r"|(паролі\s+(усіх|всіх)|api[\s-]?ключ|витягни\s+(всі|усі)\s+парол)",
    "approval_bypass": r"(without|skip|bypass)\s+(any\s+)?(approval|confirmation|human)"
                       r"|(без\s+підтвердження|не\s+питай\s+(дозволу|людину|нікого)"
                       r"|обійди\s+(перевірку|підтвердження))",
    "tool_hijack": r"(call|invoke|execute)\s+(the\s+)?tool\b"
                   r"|(виклич|запусти)\s+(інструмент|tool)\b",
    "encoded_payload": r"[A-Za-z0-9+/]{40,}={0,2}",
    "delimiter_injection": r"<\|im_start\|>|<\|system\|>|```\s*system|\[/?SYSTEM\]",
}


@dataclass
class Verdict:
    """Результат перевірки тексту на ін'єкцію."""

    blocked: bool
    rules: list[str] = field(default_factory=list)
    reason: str = ""


def detect_injection(text: str) -> Verdict:
    """Шукає ознаки prompt injection у користувацькому тексті."""
    matched = [
        name for name, pattern in INJECTION_RULES.items()
        if re.search(pattern, text or "", re.IGNORECASE)
    ]
    if not matched:
        return Verdict(blocked=False)
    return Verdict(
        blocked=True,
        rules=matched,
        reason="Запит відхилено input guardrail-ом: " + ", ".join(matched),
    )


# --- tool guardrail --------------------------------------------------------


class SearchKbArgs(BaseModel):
    query: str = Field(min_length=3, max_length=200)
    limit: int = Field(default=3, ge=1, le=5)


class EmployeeArgs(BaseModel):
    employee_id: str = Field(pattern=r"^[Ee]\d{4}$")


class TicketArgs(EmployeeArgs):
    status: Literal["open", "closed", "all"] = "open"


class ResetArgs(EmployeeArgs):
    reason: str = Field(min_length=5, max_length=200)


ARG_SCHEMAS = {
    "search_kb": SearchKbArgs,
    "lookup_employee": EmployeeArgs,
    "list_tickets": TicketArgs,
    "reset_password": ResetArgs,
}


def check_tool_call(agent: str, tool: str, args: dict) -> dict:
    """Allowlist + валідація аргументів. Повертає нормалізовані аргументи або кидає GuardrailError."""
    if tool not in ALLOWLIST.get(agent, set()):
        raise GuardrailError(f"allowlist: агенту {agent} заборонено викликати {tool}")

    schema = ARG_SCHEMAS.get(tool)
    if schema is None:
        raise GuardrailError(f"allowlist: інструмент {tool} відсутній у політиці")

    try:
        clean = schema(**args).model_dump()
    except ValidationError as error:
        problems = "; ".join(f"{e['loc'][0]}: {e['msg']}" for e in error.errors())
        raise GuardrailError(f"arguments: {tool} — {problems}") from None

    # Ін'єкція другого порядку: текст, який агент сам склав з чужих даних.
    for name, value in clean.items():
        if isinstance(value, str) and detect_injection(value).blocked:
            raise GuardrailError(f"arguments: {tool}.{name} містить ознаки ін'єкції")
    return clean


# --- output guardrail ------------------------------------------------------

PII_PATTERNS = [
    ("card", r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "[картку приховано]"),
    ("iban", r"\bUA\d{27}\b", "[IBAN приховано]"),
    ("phone", r"\+?3?8?0?[\s-]?\(?0?\d{2}\)?[\s-]?\d{3}[\s-]?\d{2}[\s-]?\d{2}\b",
     "[телефон приховано]"),
    ("email", r"[\w.+-]+@[\w-]+\.[\w.-]+\w", "[email приховано]"),
    ("password", r"(?i)((?:тимчасовий\s+пароль|temporary[_\s]?password)\W{0,4})([^\s,;\"']{8,})",
     r"\1[пароль приховано]"),
    ("tax_id", r"(?<!\d)\d{10}(?!\d)", "[ІПН приховано]"),
]


def redact_pii(text: str) -> tuple[str, list[str]]:
    """Маскує персональні дані у фінальній відповіді. Повертає текст і перелік знайдених типів."""
    found = []
    for name, pattern, replacement in PII_PATTERNS:
        text, count = re.subn(pattern, replacement, text or "")
        if count:
            found.append(name)
    return text, found


def redact_answer(answer: dict) -> tuple[dict, list[str]]:
    """Те саме, але для структурованої відповіді: обходить усі рядки всередині."""
    found: list[str] = []

    def walk(value):
        if isinstance(value, str):
            clean, names = redact_pii(value)
            found.extend(n for n in names if n not in found)
            return clean
        if isinstance(value, list):
            return [walk(v) for v in value]
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        return value

    return walk(answer), found
