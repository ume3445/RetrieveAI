from __future__ import annotations

from dataclasses import dataclass

import chromadb
from openai import OpenAI

from . import config


@dataclass
class Chunk:
    text: str
    score: float  # cosine similarity; higher is more relevant
    page: int
    source: str


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def retrieve(
    collection: chromadb.Collection,
    query: str,
    top_k: int | None = None,
) -> list[Chunk]:
    """
    ChromaDB's HNSW index returns approximate neighbours. We fetch top_k*3
    candidates then re-score with exact cosine to correct for approximation drift.
    """
    k = top_k if top_k is not None else config.TOP_K
    fetch_k = min(k * 3, collection.count())
    if fetch_k == 0:
        return []

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    query_vec = client.embeddings.create(model=config.EMBED_MODEL, input=[query]).data[0].embedding

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=fetch_k,
        include=["documents", "embeddings", "metadatas"],
    )

    chunks = [
        Chunk(
            text=doc,
            score=_cosine(query_vec, emb),
            page=meta.get("page", 0),
            source=meta.get("source", ""),
        )
        for doc, emb, meta in zip(
            results["documents"][0],
            results["embeddings"][0],
            results["metadatas"][0],
        )
    ]

    chunks.sort(key=lambda c: c.score, reverse=True)
    return chunks[:k]
