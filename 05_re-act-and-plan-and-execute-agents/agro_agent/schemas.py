"""Structured outputs: план, рішення replanner-а та фінальна відповідь.

Field(description=...) обов'язковий: саме з цих описів модель розуміє, що заповнювати.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Plan(BaseModel):
    """План робіт: ціль і послідовність конкретних кроків."""

    goal: str = Field(description="Головна ціль задачі одним реченням")
    steps: list[str] = Field(
        description="2-5 конкретних кроків. Кожен крок — одна дія, яку можна виконати інструментом"
    )


class ReplanDecision(BaseModel):
    """Рішення replanner-а після виконання чергового кроку."""

    action: Literal["continue", "replan", "finish"] = Field(
        description=(
            "continue = виконати наступний крок плану, "
            "replan = замінити кроки, що залишилися, "
            "finish = ціль досягнута або далі рухатися немає сенсу"
        )
    )
    updated_steps: Optional[list[str]] = Field(
        default=None, description="Нові кроки замість тих, що залишилися (лише коли action=replan)"
    )
    reasoning: str = Field(description="Коротке пояснення рішення українською")


class AgroAnswer(BaseModel):
    """Фінальна відповідь агронома."""

    summary: str = Field(description="Коротка відповідь українською, 3-6 речень")
    facts: list[str] = Field(
        default_factory=list, description="Ключові числа з інструментів і бази знань"
    )
    actions: list[str] = Field(
        default_factory=list,
        description="Виконані незворотні дії (наряд на обробку) або позначка, що їх відхилено",
    )
    warnings: list[str] = Field(
        default_factory=list, description="Ризики, чого дізнатися не вдалося, що варто перевірити"
    )
