"""Інструменти MCP-сервера: логіка чиста, тож тести офлайн і детерміновані."""

import pytest

import mcp_server


def test_search_kb_finds_relevant_article():
    found = mcp_server.search_kb("заблоковано обліковий запис пароль")
    assert found, "по ключових словах має знайтись хоча б одна стаття"
    assert found[0]["article_id"] in {"KB-01", "KB-02"}
    assert all(a["score"] > 0 for a in found)


def test_search_kb_returns_nothing_for_unrelated_query():
    assert mcp_server.search_kb("кава молоко печиво") == []


def test_lookup_employee_is_case_insensitive_and_reports_unknown():
    assert mcp_server.lookup_employee("e1001")["name"] == "Олена Ковальчук"
    assert "error" in mcp_server.lookup_employee("E9999")


def test_list_tickets_filters_by_status():
    assert [t["ticket_id"] for t in mcp_server.list_tickets("E1001")] == ["T-2041"]
    assert len(mcp_server.list_tickets("E1001", "all")) == 2
    assert mcp_server.list_tickets("E1002", "closed") == []


def test_reset_password_unlocks_account_and_returns_temporary_password():
    mcp_server.EMPLOYEES["E1001"]["account_locked"] = True
    result = mcp_server.reset_password("E1001", "заявка T-2041")

    assert len(result["temporary_password"]) >= 8
    assert result["must_change_on_login"] is True
    assert mcp_server.EMPLOYEES["E1001"]["account_locked"] is False
    assert "error" in mcp_server.reset_password("E9999", "будь-яка причина")


@pytest.mark.asyncio
async def test_all_four_tools_are_registered_in_the_server():
    registered = await mcp_server.mcp.list_tools()
    assert {t.name for t in registered} == {
        "search_kb", "lookup_employee", "list_tickets", "reset_password",
    }
    # Опис читає LLM під час вибору інструмента — порожній опис ламає добір.
    assert all(tool.description for tool in registered)
