from __future__ import annotations

import re
from dataclasses import dataclass

import chromadb
from openai import OpenAI
from rank_bm25 import BM25Okapi

from . import config

# Weight given to the dense (cosine) score in hybrid_retrieve's combination;
# the remainder (1 - DENSE_WEIGHT) goes to the BM25 score.
DENSE_WEIGHT = 0.7


@dataclass
class Chunk:
    text: str
    score: float  # cosine similarity (retrieve) or combined score (hybrid_retrieve)
    page: int
    source: str


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _min_max_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


@dataclass
class _BM25Index:
    bm25: BM25Okapi
    ids: list[str]  # corpus order, matches what bm25.get_scores() returns
    documents: dict[str, str]
    metadatas: dict[str, dict]


_bm25_cache: dict[str, _BM25Index] = {}


def _get_bm25_index(collection: chromadb.Collection) -> _BM25Index:
    """Build (and cache, per collection name) an in-memory BM25 index over all chunks."""
    cached = _bm25_cache.get(collection.name)
    if cached is not None:
        return cached

    result = collection.get(include=["documents", "metadatas"])
    ids = result["ids"]
    tokenized = [_tokenize(doc) for doc in result["documents"]]

    index = _BM25Index(
        bm25=BM25Okapi(tokenized),
        ids=ids,
        documents=dict(zip(ids, result["documents"])),
        metadatas=dict(zip(ids, result["metadatas"])),
    )
    _bm25_cache[collection.name] = index
    return index


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


def hybrid_retrieve(
    collection: chromadb.Collection,
    query: str,
    top_k: int | None = None,
) -> list[Chunk]:
    """
    Independent candidate generation from both retrieval methods, unioned
    before ranking — so BM25 can surface a chunk dense retrieval missed
    entirely, and vice versa. Dense candidates come from HNSW + exact cosine
    rescore (top_k*3, same as retrieve()); BM25 candidates come from scoring
    every chunk in the collection via the cached BM25Okapi index and taking
    the top top_k*3 by BM25 score. A chunk absent from one side scores 0 on
    that side. Both score types are then min-max normalized across the union
    (not their original individual sets) and combined as a DENSE_WEIGHT-
    weighted sum, then re-ranked.
    """
    k = top_k if top_k is not None else config.TOP_K
    fetch_k = min(k * 3, collection.count())
    if fetch_k == 0:
        return []

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    query_vec = client.embeddings.create(model=config.EMBED_MODEL, input=[query]).data[0].embedding

    dense_results = collection.query(
        query_embeddings=[query_vec],
        n_results=fetch_k,
        include=["embeddings"],
    )
    dense_ids = dense_results["ids"][0]
    dense_score_by_id = {
        cid: _cosine(query_vec, emb)
        for cid, emb in zip(dense_ids, dense_results["embeddings"][0])
    }

    bm25_index = _get_bm25_index(collection)
    bm25_full_scores = bm25_index.bm25.get_scores(_tokenize(query)).tolist()
    bm25_ranked = sorted(zip(bm25_index.ids, bm25_full_scores), key=lambda p: p[1], reverse=True)
    bm25_score_by_id = dict(bm25_ranked[:fetch_k])

    union_ids = list(dict.fromkeys([*dense_ids, *bm25_score_by_id.keys()]))
    dense_norm = _min_max_normalize([dense_score_by_id.get(cid, 0.0) for cid in union_ids])
    bm25_norm = _min_max_normalize([bm25_score_by_id.get(cid, 0.0) for cid in union_ids])

    chunks = [
        Chunk(
            text=bm25_index.documents[cid],
            score=DENSE_WEIGHT * dense_norm[i] + (1 - DENSE_WEIGHT) * bm25_norm[i],
            page=bm25_index.metadatas[cid].get("page", 0),
            source=bm25_index.metadatas[cid].get("source", ""),
        )
        for i, cid in enumerate(union_ids)
    ]

    chunks.sort(key=lambda c: c.score, reverse=True)
    return chunks[:k]
