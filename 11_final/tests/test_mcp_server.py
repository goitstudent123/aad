"""Unit-тести MCP-сервера: tools, resources, prompts. Офлайн, без ключа й мережі."""

import json

import pytest

from mcp_server import mcp


async def call(name: str, args: dict) -> dict:
    """Хелпер: виклик MCP-tool у тестах (FastMCP віддає (blocks, meta))."""
    result = await mcp.call_tool(name, args)
    blocks = result[0] if isinstance(result, tuple) else result
    return json.loads(blocks[0].text)


async def test_list_tools_registers_all_five():
    names = {t.name for t in await mcp.list_tools()}
    assert {"get_ticket", "search_tickets", "get_customer",
            "get_billing_summary", "update_ticket_status"} <= names


async def test_every_tool_has_a_docstring():
    # Опис інструмента читає LLM — порожній опис ламає вибір інструмента.
    for tool in await mcp.list_tools():
        assert tool.description and len(tool.description) > 40, tool.name


async def test_get_ticket_found_and_not_found():
    found = await call("get_ticket", {"ticket_id": "tkt-001"})  # регістр не має значення
    assert found["id"] == "TKT-001" and found["customer_id"] == "C-100"
    assert "error" in await call("get_ticket", {"ticket_id": "TKT-999"})


async def test_search_tickets_filters_by_status():
    every = await call("search_tickets", {"customer_id": "C-100", "status": "all"})
    resolved = await call("search_tickets", {"customer_id": "C-100", "status": "resolved"})
    assert every["count"] == 2
    assert resolved["count"] == 1 and resolved["tickets"][0]["id"] == "TKT-003"
    assert "error" in await call("search_tickets", {"customer_id": "C-100", "status": "BOGUS"})


async def test_get_customer_returns_pii_fields():
    customer = await call("get_customer", {"customer_id": "C-100"})
    assert customer["name"] == "Олег Петренко"
    assert "@" in customer["email"] and customer["phone"].startswith("+380")
    assert "error" in await call("get_customer", {"customer_id": "C-999"})


async def test_billing_summary_counts_balance_and_validates_period():
    summary = await call("get_billing_summary", {"customer_id": "C-100", "period": "2026-09"})
    assert summary["charged"] == 450 and summary["paid"] == 0 and summary["balance"] == -450
    assert "error" in await call("get_billing_summary", {"customer_id": "C-100",
                                                        "period": "вересень"})
    assert "error" in await call("get_billing_summary", {"customer_id": "C-100",
                                                         "period": "2019-01"})


async def test_update_ticket_status_valid_and_invalid():
    updated = await call("update_ticket_status", {"ticket_id": "TKT-002",
                                                  "new_status": "resolved",
                                                  "reason": "unit test"})
    assert updated["updated"] == "TKT-002" and updated["new_status"] == "resolved"
    assert updated["old_status"] == "in_progress"

    assert "error" in await call("update_ticket_status", {"ticket_id": "TKT-002",
                                                          "new_status": "BOGUS"})
    assert "error" in await call("update_ticket_status", {"ticket_id": "TKT-999",
                                                          "new_status": "closed"})


async def test_list_resources_contains_faq_and_policy():
    uris = {str(r.uri) for r in await mcp.list_resources()}
    assert "faq://general" in uris and "policy://refund" in uris


async def test_resource_returns_readable_content():
    faq = json.loads((await mcp.read_resource("faq://general"))[0].content)
    assert any("повернення" in item["q"].lower() for item in faq)


async def test_list_prompts_and_render_support_reply():
    prompts = {p.name for p in await mcp.list_prompts()}
    assert {"support_reply", "escalation_note"} <= prompts

    rendered = await mcp.get_prompt("support_reply", {"customer_name": "Олег Петренко",
                                                      "issue_summary": "не списано платіж",
                                                      "tone": "empathetic"})
    text = rendered.messages[0].content.text
    assert "Олег Петренко" in text and "теплу" in text


@pytest.mark.parametrize("tool", ["get_ticket", "get_customer"])
async def test_tools_do_not_crash_on_garbage_input(tool):
    key = "ticket_id" if tool == "get_ticket" else "customer_id"
    assert "error" in await call(tool, {key: "'; DROP TABLE tickets --"})
