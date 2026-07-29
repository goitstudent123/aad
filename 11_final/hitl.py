"""HITL: approval gate для ризикових MCP-інструментів — approve / reject / edit.

Пауза стається ДО звернення до MCP, тож незворотна дія фізично не виконується без
рішення людини. Стан лежить у SqliteSaver, тому рішення може прийти з іншого процесу.
"""

import asyncio
from dataclasses import dataclass, field

from langgraph.types import Command, interrupt

from config import RISKY_TOOLS
from logs import short, trace


@dataclass
class Decision:
    """Нормалізоване рішення людини."""

    action: str  # approve | reject | edit
    args: dict = field(default_factory=dict)
    comment: str = ""

    @property
    def approved(self) -> bool:
        return self.action in ("approve", "edit")


def normalise(raw, args: dict) -> Decision:
    """Рішення приймаємо і як bool, і як {'approved': ...}, і як {'action': ...}."""
    if isinstance(raw, bool):
        return Decision("approve" if raw else "reject", dict(args))
    if not isinstance(raw, dict):
        return Decision("reject", dict(args), "невідомий формат рішення")

    action = raw.get("action")
    if action is None:
        action = "approve" if raw.get("approved") else "reject"
    merged = {**args, **(raw.get("args") or {})}
    if action == "edit" and not (raw.get("args") or {}):
        action = "approve"
    return Decision(action, merged, raw.get("comment", "") or raw.get("reason", ""))


def approval_gate(tool: str, agent: str, args: dict) -> Decision:
    """Зупиняє граф на ризиковому виклику й повертає рішення людини."""
    if tool not in RISKY_TOOLS:
        return Decision("approve", dict(args))

    trace("hitl", f"⏸ чекає підтвердження: {tool}({short(args, 120)})")
    decision = normalise(
        interrupt({
            "message": "Підтвердити ризикову дію",
            "tool": tool,
            "agent_name": agent,
            "args": args,
            "options": ["approve", "reject", "edit"],
        }),
        args,
    )
    trace("hitl", f"рішення людини: {decision.action} {short(decision.args, 100)}")
    return decision


def resume_command(action: str, args: dict | None = None, comment: str = "") -> Command:
    """Command для продовження графа після паузи."""
    return Command(resume={"action": action, "args": args or {}, "comment": comment})


# ── Демонстрація трьох сценаріїв ───────────────────────────────────────────

RISKY_QUERY = "Закрий тікет TKT-001 — клієнт підтвердив, гроші повернуто."


async def _scenario(app, thread: str, action: str, args: dict | None = None, comment: str = ""):
    """Один сценарій: запуск до паузи → рішення людини → продовження."""
    # Імпорт усередині: mas_langgraph тягне цей модуль через mcp_client.
    from mas_langgraph import result_of, run_mas

    config = {"configurable": {"thread_id": thread}}
    first = await run_mas(app, RISKY_QUERY, config)
    if not first["pending_approval"]:
        return {"scenario": action, "paused": False, **first}

    trace("hitl", f"сценарій {action}: пауза на {first['pending_approval'][0]['tool']}")
    state = await app.ainvoke(resume_command(action, args, comment),
                              {**config, "recursion_limit": 40})
    return {"scenario": action, "paused": True, "pending_before": first["pending_approval"],
            **result_of(state)}


async def run_scenarios() -> dict:
    """approve / reject / edit на одному й тому самому ризиковому інструменті."""
    from config import make_saver
    from mas_langgraph import build_default

    out = {}
    async with make_saver() as saver:
        app = await build_default(checkpointer=saver)
        out["approve"] = await _scenario(app, "hitl-approve", "approve")
        out["reject"] = await _scenario(app, "hitl-reject", "reject",
                                        comment="клієнт не підтвердив повернення")
        out["edit"] = await _scenario(app, "hitl-edit", "edit",
                                      args={"new_status": "resolved",
                                            "reason": "виправлено людиною: не closed, а resolved"})
    return out


if __name__ == "__main__":
    print(asyncio.run(run_scenarios()))
