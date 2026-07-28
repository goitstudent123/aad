"""Тести конфігурації клієнтів: мережевих викликів тут немає, лише параметри об'єктів."""

from plan_agent import config


def test_llm_client_has_timeout_and_retries(monkeypatch):
    """Без тайм-аута httpx чекає на зависший провайдер вічно — і агент німо висне."""
    monkeypatch.setenv(config.API_KEY_ENV, "test-key")

    llm = config.make_llm()

    assert llm.request_timeout == config.LLM_TIMEOUT
    assert llm.max_retries == config.LLM_RETRIES


def test_checkpointer_uses_a_file_not_memory(tmp_path):
    """Persistence між процесами працює лише на файлі."""
    path = tmp_path / "agent_state.db"

    saver = config.make_saver(str(path))

    assert path.exists()
    assert config.DB_PATH.endswith(".db") and ":memory:" not in config.DB_PATH
    saver.conn.close()
