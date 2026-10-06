from src import workflow
from src.retrieval import Chunk


def test_app_retrieval_uses_hybrid(monkeypatch):
    calls = []

    def fake_hybrid(collection, query, top_k=None):
        calls.append((query, top_k))
        return [Chunk(text="t", score=0.03, page=1, source="s")]

    monkeypatch.setattr(workflow, "hybrid_retrieve", fake_hybrid)
    state = workflow._node_retrieve({"query": "q", "collection": None, "top_k": 5, "chunks": [], "answer": None})
    assert calls == [("q", 5)]
    assert len(state["chunks"]) == 1
