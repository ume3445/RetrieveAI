from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import chromadb
from openai import OpenAI
from rank_bm25 import BM25Okapi

from . import config

# Reciprocal Rank Fusion constant. 60 is the value from Cormack et al. (2009)
# and the common default; it was not tuned on the eval set.
RRF_K = 60


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
    # NFKC here too, so queries (and any chunks stored before ingestion-time
    # cleaning existed) tokenize the same way as cleaned chunks.
    return re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", text).lower())


def _rrf_fuse(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """
    Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank(d)).
    Works on ranks, not raw scores, so BM25 scores (unbounded) and cosine
    similarities (clustered near each other) never need to be put on the same
    scale, and a chunk that only one retriever found keeps the credit it earned
    there instead of being scored as 0 by the other.
    """
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank)
    return fused


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
    Dense and BM25 retrieval each produce an independent top top_k*3 ranking
    (dense: HNSW candidates re-ranked by exact cosine; BM25: every chunk scored,
    zero-score chunks dropped). The two rankings are merged with Reciprocal
    Rank Fusion, so a chunk only BM25 found can still outrank one only dense
    found. Chunk.score is the RRF score.
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
    dense_scored = [
        (cid, _cosine(query_vec, emb))
        for cid, emb in zip(dense_results["ids"][0], dense_results["embeddings"][0])
    ]
    dense_ranking = [cid for cid, _ in sorted(dense_scored, key=lambda p: p[1], reverse=True)]

    bm25_index = _get_bm25_index(collection)
    bm25_scores = bm25_index.bm25.get_scores(_tokenize(query)).tolist()
    bm25_ranked = sorted(zip(bm25_index.ids, bm25_scores), key=lambda p: p[1], reverse=True)
    bm25_ranking = [cid for cid, score in bm25_ranked[:fetch_k] if score > 0]

    fused = _rrf_fuse([dense_ranking, bm25_ranking])
    ranked_ids = sorted(fused, key=lambda cid: fused[cid], reverse=True)[:k]

    return [
        Chunk(
            text=bm25_index.documents[cid],
            score=fused[cid],
            page=bm25_index.metadatas[cid].get("page", 0),
            source=bm25_index.metadatas[cid].get("source", ""),
        )
        for cid in ranked_ids
    ]
