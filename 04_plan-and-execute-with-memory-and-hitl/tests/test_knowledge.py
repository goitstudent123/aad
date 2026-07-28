"""Тести RAG-інструмента: ні мережі, ні ChromaDB — колекція та embeddings підмінені."""

import pytest
from pydantic import ValidationError

from plan_agent import knowledge
from plan_agent.knowledge import SearchArgs


class FakeCollection:
    """Мінімальна заміна колекції Chroma: віддає те, що їй сказали віддати."""

    def __init__(self, documents):
        self.documents = documents
        self.queries = []

    def query(self, query_embeddings, n_results):
        self.queries.append((query_embeddings, n_results))
        return {"documents": [self.documents[:n_results]]}


@pytest.fixture
def fake_store(monkeypatch):
    def _install(documents):
        store = FakeCollection(documents)
        monkeypatch.setattr(knowledge, "collection", lambda: store)
        monkeypatch.setattr(knowledge, "embed", lambda texts: [[0.1, 0.2] for _ in texts])
        return store

    return _install


def test_knowledge_base_has_at_least_eight_documents():
    assert len(knowledge.DOCUMENTS) >= 8
    assert len(knowledge.DOC_IDS) == len(set(knowledge.DOC_IDS)) == len(knowledge.DOCUMENTS)
    # Документи мають бути змістовними, а не рядком на три слова.
    assert all(len(doc) > 150 for doc in knowledge.DOCUMENTS)


def test_search_returns_top_documents_joined(fake_store):
    store = fake_store(["перший документ", "другий документ", "третій документ", "четвертий"])

    result = knowledge.search_knowledge.invoke({"query": "страховка Шенген"})

    assert result == "перший документ\n---\nдругий документ\n---\nтретій документ"
    assert store.queries[0][1] == 3


def test_search_respects_n_results(fake_store):
    store = fake_store(["а", "б", "в", "г", "д"])

    knowledge.search_knowledge.invoke({"query": "візи", "n_results": 1})

    assert store.queries[0][1] == 1


def test_search_reports_empty_base(fake_store):
    fake_store([])
    assert "error" in knowledge.search_knowledge.invoke({"query": "будь-що"})


def test_search_reports_broken_store(monkeypatch):
    def _boom():
        raise RuntimeError("chroma лежить")

    monkeypatch.setattr(knowledge, "collection", _boom)
    monkeypatch.setattr(knowledge, "embed", lambda texts: [[0.0]])
    assert "недоступна" in knowledge.search_knowledge.invoke({"query": "візи"})


@pytest.mark.parametrize("args", [{"query": ""}, {"query": "візи", "n_results": 0},
                                 {"query": "візи", "n_results": 9}])
def test_search_args_are_validated(args):
    with pytest.raises(ValidationError):
        SearchArgs(**args)
