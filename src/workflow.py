from __future__ import annotations

from typing import TypedDict

import chromadb
from langgraph.graph import END, StateGraph

from . import config
from .generation import Answer, generate
from .retrieval import Chunk, retrieve


class RAGState(TypedDict):
    query: str
    collection: chromadb.Collection
    top_k: int
    chunks: list[Chunk]
    answer: Answer | None


def _node_retrieve(state: RAGState) -> RAGState:
    chunks = retrieve(state["collection"], state["query"], top_k=state["top_k"])
    return {**state, "chunks": chunks}


def _node_rerank(state: RAGState) -> RAGState:
    chunks = state["chunks"]
    if not chunks:
        return state
    # Drop chunks that score below half the top result; keeps context tight
    # without an absolute threshold that varies across queries.
    floor = chunks[0].score * 0.5
    filtered = [c for c in chunks if c.score >= floor]
    return {**state, "chunks": filtered or chunks}


def _node_generate(state: RAGState) -> RAGState:
    return {**state, "answer": generate(state["query"], state["chunks"])}


def _build_graph():
    g = StateGraph(RAGState)
    g.add_node("retrieve", _node_retrieve)
    g.add_node("rerank", _node_rerank)
    g.add_node("generate", _node_generate)
    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "generate")
    g.add_edge("generate", END)
    return g.compile()


_graph = _build_graph()


def run(collection: chromadb.Collection, query: str, top_k: int | None = None) -> Answer:
    state: RAGState = _graph.invoke(
        {
            "query": query,
            "collection": collection,
            "top_k": top_k if top_k is not None else config.TOP_K,
            "chunks": [],
            "answer": None,
        }
    )
    return state["answer"]
